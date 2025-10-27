from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source.")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources."""

    answer: str = Field(description="The answer.")
    source: list[Source] = Field(
        default_factory=list, description="The source used by the agent."
    )
