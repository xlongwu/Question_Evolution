import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class SchemaValidationError(ValueError):
    pass


def load_schema(schema_path: Path) -> Dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    if not isinstance(schema, dict):
        raise SchemaValidationError(f"{schema_path} is not a JSON object schema")
    return schema


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _resolve_ref(ref: str, schema_dir: Path, cache: Dict[Path, Dict[str, Any]]) -> Dict[str, Any]:
    ref_path = (schema_dir / ref).resolve()
    if ref_path not in cache:
        cache[ref_path] = load_schema(ref_path)
    return cache[ref_path]


def validate_instance(
    value: Any,
    schema: Dict[str, Any],
    *,
    schema_dir: Optional[Path] = None,
    path: str = "$",
    cache: Optional[Dict[Path, Dict[str, Any]]] = None,
) -> None:
    schema_dir = schema_dir or Path("schemas").resolve()
    cache = cache if cache is not None else {}

    if "$ref" in schema:
        ref_schema = _resolve_ref(str(schema["$ref"]), schema_dir, cache)
        validate_instance(value, ref_schema, schema_dir=schema_dir, path=path, cache=cache)
        return

    if "type" in schema:
        expected_types = schema["type"]
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if isinstance(expected_types, list) and not any(_type_matches(value, str(item)) for item in expected_types):
            raise SchemaValidationError(f"{path} expected type {expected_types}, got {type(value).__name__}")

    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path} expected one of {schema['enum']!r}, got {value!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path} below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path} above maximum {schema['maximum']}")

    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        raise SchemaValidationError(f"{path} shorter than minLength {schema['minLength']}")

    if isinstance(value, dict):
        for field in schema.get("required", []):
            if field not in value:
                raise SchemaValidationError(f"{path}.{field} is required")
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for field, field_schema in properties.items():
                if field in value and isinstance(field_schema, dict):
                    validate_instance(
                        value[field],
                        field_schema,
                        schema_dir=schema_dir,
                        path=f"{path}.{field}",
                        cache=cache,
                    )

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for idx, item in enumerate(value):
            validate_instance(
                item,
                schema["items"],
                schema_dir=schema_dir,
                path=f"{path}[{idx}]",
                cache=cache,
            )


def validate_records_against_schema(
    records: Iterable[Dict[str, Any]],
    schema_path: Path,
) -> List[SchemaValidationError]:
    schema_path = schema_path.resolve()
    schema = load_schema(schema_path)
    errors: List[SchemaValidationError] = []
    for idx, record in enumerate(records):
        try:
            validate_instance(record, schema, schema_dir=schema_path.parent, path=f"$[{idx}]")
        except SchemaValidationError as exc:
            errors.append(exc)
    return errors
