"""Compatibility helpers for running the repo on both Pydantic v1 and v2."""

from __future__ import annotations

import json

from pydantic import BaseModel


if not hasattr(BaseModel, "model_validate"):
    @classmethod
    def _model_validate(cls, obj):
        return cls.parse_obj(obj)

    def _model_dump(self, *, mode: str | None = None, **kwargs):
        del mode
        return self.dict(**kwargs)

    def _model_dump_json(self, *, indent: int | None = None, ensure_ascii: bool = False, **kwargs):
        payload = self.dict(**kwargs)
        return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)

    def _model_copy(self, *, deep: bool = False, update: dict | None = None):
        return self.copy(deep=deep, update=update)

    BaseModel.model_validate = _model_validate
    BaseModel.model_dump = _model_dump
    BaseModel.model_dump_json = _model_dump_json
    BaseModel.model_copy = _model_copy
