from pydantic import BaseModel, Field, field_validator, model_validator


class AdmissionDecision(BaseModel):
    # BenchTools uses this schema to reject malformed model outputs.
    admit: list[str] = Field(min_length=4, max_length=4)
    ranking: list[str] = Field(min_length=10, max_length=10)

    @field_validator("admit", "ranking")
    @classmethod
    def patient_ids_must_be_strings(cls, patient_ids: list[str]) -> list[str]:
        # Keep validation format-focused; the analyzer checks episode membership.
        for patient_id in patient_ids:
            if not isinstance(patient_id, str) or not patient_id.startswith("P"):
                raise ValueError("patient IDs must be strings like P1")
        return patient_ids

    @model_validator(mode="after")
    def admitted_patients_must_lead_ranking(self) -> "AdmissionDecision":
        # This catches contradictions like admit=["P1"] but ranking starts ["P2"].
        if len(set(self.admit)) != len(self.admit):
            raise ValueError("admit must not contain duplicate patient IDs")
        if len(set(self.ranking)) != len(self.ranking):
            raise ValueError("ranking must not contain duplicate patient IDs")
        if self.ranking[: len(self.admit)] != self.admit:
            raise ValueError("admit must match the first patients in ranking")
        return self
