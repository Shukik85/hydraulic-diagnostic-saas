"""
CSV Upload & Validation API
Level 6B Option A: Upload historical data
"""

import io
import uuid
from datetime import UTC, datetime

import pandas as pd
from db.session import get_db
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from middleware.auth import get_current_user
from models.equipment import Equipment
from models.sensor_data import SensorData
from models.sensor_mapping import DataSource, SensorMapping
from schemas.csv_upload import CSVImportResponse, CSVValidationResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/csv-upload", tags=["CSV Upload"])


@router.post("/validate", response_model=CSVValidationResponse)
async def validate_csv(
    equipment_id: uuid.UUID,
    file: UploadFile = File(...),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
    current_user=Depends(get_current_user),  # noqa: B008
):
    """
    Validate CSV file before import
    Checks: timestamp format, sensor IDs, value ranges
    """
    equipment = await db.get(Equipment, equipment_id)
    if not equipment or equipment.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Equipment not found")

    df = await _read_csv(file)
    errors, warnings = [], []

    _validate_timestamp_column(df, errors)
    sensor_mappings = await _get_sensor_mappings(db, equipment_id, errors)
    if not sensor_mappings:
        return {"valid": False, "errors": errors, "warnings": [], "stats": {}}

    valid_sensor_ids = set(sensor_mappings.keys())
    csv_columns = set(df.columns) - {"timestamp"}

    _check_missing_and_unknown_sensors(valid_sensor_ids, csv_columns, errors, warnings)
    _check_value_ranges(df, csv_columns, valid_sensor_ids, sensor_mappings, warnings)
    _check_duplicates(df, warnings)

    stats = _calculate_stats(df, csv_columns, valid_sensor_ids)

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "preview": df.head(10).to_dict("records") if len(df) > 0 else [],
    }


async def _read_csv(file: UploadFile):
    try:
        contents = await file.read()
        return pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid CSV format: {str(e)}"
        ) from e


def _validate_timestamp_column(df, errors):
    if "timestamp" not in df.columns:
        errors.append({"field": "timestamp", "message": "Missing timestamp column"})
    else:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            errors.append(
                {"field": "timestamp", "message": f"Invalid datetime format: {str(e)}"}
            )


async def _get_sensor_mappings(db, equipment_id, errors):
    from sqlalchemy import select

    result = await db.execute(
        select(SensorMapping).where(SensorMapping.equipment_id == equipment_id)
    )
    sensor_mappings = {sm.sensor_id: sm for sm in result.scalars().all()}
    if not sensor_mappings:
        errors.append(
            {
                "field": "sensor_mappings",
                "message": "No sensors mapped to equipment. Complete Level 6A first.",
            }
        )
    return sensor_mappings


def _check_missing_and_unknown_sensors(valid_sensor_ids, csv_columns, errors, warnings):
    if missing_sensors := valid_sensor_ids - csv_columns:
        warnings.append(
            {
                "field": "sensors",
                "message": f"Missing data for sensors: {', '.join(missing_sensors)}",
            }
        )

    if unknown_sensors := csv_columns - valid_sensor_ids:
        errors.append(
            {
                "field": "sensors",
                "message": f"Unknown sensors (not mapped): {', '.join(unknown_sensors)}",
            }
        )


def _check_value_ranges(df, csv_columns, valid_sensor_ids, sensor_mappings, warnings):
    for sensor_id in csv_columns & valid_sensor_ids:
        mapping = sensor_mappings[sensor_id]
        if mapping.expected_range_min and mapping.expected_range_max:
            values = df[sensor_id].dropna()
            out_of_range = values[
                (values < mapping.expected_range_min)
                | (values > mapping.expected_range_max)
            ]
            if len(out_of_range) > 0:
                percentage = len(out_of_range) / len(values) * 100
                warnings.append(
                    {
                        "sensor": sensor_id,
                        "message": f"{len(out_of_range)} values ({percentage:.1f}%) out of expected range [{mapping.expected_range_min}, {mapping.expected_range_max}]",
                        "severity": "high" if percentage > 10 else "medium",
                    }
                )


def _check_duplicates(df, warnings):
    if "timestamp" in df.columns:
        duplicates = df[df.duplicated(subset=["timestamp"], keep=False)]
        if len(duplicates) > 0:
            warnings.append(
                {
                    "field": "timestamp",
                    "message": f"{len(duplicates)} duplicate timestamps found",
                }
            )


def _calculate_stats(df, csv_columns, valid_sensor_ids):
    stats = {
        "rows_total": len(df),
        "rows_valid": len(df.dropna()),
        "rows_invalid": len(df) - len(df.dropna()),
        "sensors_count": len(csv_columns & valid_sensor_ids),
        "date_range": {},
    }
    if "timestamp" in df.columns and len(df) > 0:
        stats["date_range"] = {
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat(),
            "duration_hours": (
                df["timestamp"].max() - df["timestamp"].min()
            ).total_seconds()
            / 3600,
        }
    return stats


@router.post("/import", response_model=CSVImportResponse)
async def import_csv(
    equipment_id: uuid.UUID,
    file: UploadFile = File(...),  # noqa: B008
    ignore_warnings: bool = False,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    current_user=Depends(get_current_user),  # noqa: B008
):
    """
    Import CSV data into database
    """
    # First validate
    validation = await validate_csv(equipment_id, file, db, current_user)

    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={"message": "Validation failed", "errors": validation["errors"]},
        )

    if validation["warnings"] and not ignore_warnings:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Validation warnings found. Set ignore_warnings=true to proceed.",
                "warnings": validation["warnings"],
            },
        )

    # Read CSV again
    await file.seek(0)
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Get equipment
    equipment = await db.get(Equipment, equipment_id)

    # Get sensor mappings
    result = await db.execute(
        select(SensorMapping).where(SensorMapping.equipment_id == equipment_id)
    )
    sensor_mappings = {sm.sensor_id: sm for sm in result.scalars().all()}

    # Transform to sensor readings
    readings = []

    for _, row in df.iterrows():
        timestamp = row["timestamp"]

        for sensor_id in df.columns:
            if sensor_id == "timestamp" or sensor_id not in sensor_mappings:
                continue

            value = row[sensor_id]
            if pd.isna(value):
                continue

            mapping = sensor_mappings[sensor_id]

            reading = SensorData(
                system_id=equipment.system_id,
                sensor_id=sensor_id,
                timestamp=timestamp,
                value=float(value),
                unit=mapping.unit,
                is_valid=True,
                is_quarantined=False,
            )
            readings.append(reading)

    # Batch insert
    db.add_all(readings)

    # Create data source record
    data_source = DataSource(
        equipment_id=equipment_id,
        source_type="csv_upload",
        source_name=file.filename,
        config={
            "filename": file.filename,
            "rows": len(df),
            "sensors": list(sensor_mappings.keys()),
        },
        is_active=True,
        last_sync=datetime.now(UTC),
        total_readings=len(readings),
    )
    db.add(data_source)

    await db.commit()

    return {
        "success": True,
        "equipment_id": equipment_id,
        "data_source_id": data_source.id,
        "imported_readings": len(readings),
        "date_range": validation["stats"]["date_range"],
    }


@router.get("/template")
async def get_csv_template(
    equipment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    current_user=Depends(get_current_user),  # noqa: B008
):
    """
    Generate CSV template with sensor columns
    """
    equipment = await db.get(Equipment, equipment_id)
    if not equipment or equipment.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Equipment not found")

    # Get sensor mappings
    result = await db.execute(
        select(SensorMapping).where(SensorMapping.equipment_id == equipment_id)
    )
    sensor_mappings = result.scalars().all()

    if not sensor_mappings:
        raise HTTPException(
            status_code=400, detail="No sensors mapped. Complete Level 6A first."
        )

    # Create template
    columns = ["timestamp"] + [sm.sensor_id for sm in sensor_mappings]
    df = pd.DataFrame(columns=columns)

    # Add example rows
    # Add example rows
    example_timestamp = datetime.now().astimezone()
    example_row = {"timestamp": example_timestamp.isoformat()}
    for sm in sensor_mappings:
        example_row[sm.sensor_id] = (sm.expected_range_min + sm.expected_range_max) / 2

    df = pd.DataFrame([example_row])

    # Return as CSV
    output = io.StringIO()
    df.to_csv(output, index=False)

    return {
        "content": output.getvalue(),
        "filename": f"{equipment.system_id}_template.csv",
    }
