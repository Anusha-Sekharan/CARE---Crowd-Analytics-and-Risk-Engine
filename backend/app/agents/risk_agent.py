import logging
from enum import Enum
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# --- Schemas ---

class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskInput(BaseModel):
    people_count: int = Field(..., description="Total people count from Agent 1")
    density_score: float = Field(..., description="Overall density score from Agent 1 (0.0 to 1.0)")
    max_zone_count: int = Field(..., description="People count in the hotspot zone from Agent 2")

class RiskOutput(BaseModel):
    risk_level: RiskLevel = Field(..., description="Determined safety risk level")
    congestion_score: int = Field(..., description="Calculated congestion score (0-100)")
    hotspot_concentration: float = Field(..., description="Percentage of people concentrated in the hotspot (0.0 to 1.0)")

# --- Agent Implementation ---

class RiskAnalysisAgent:
    """
    Agent 3: Risk Analysis Agent
    Analyzes density scores and spatial hotspots to calculate an aggregate congestion score
    and categorize the immediate safety risk level.
    """
    
    def __init__(self, density_weight: float = 0.6, hotspot_weight: float = 0.4):
        """
        Initializes the agent with weighting parameters.
        
        Args:
            density_weight: Importance weight of overall density (default 0.6).
            hotspot_weight: Importance weight of localized concentration (default 0.4).
        """
        self.density_weight = density_weight
        self.hotspot_weight = hotspot_weight

    def _calculate_hotspot_concentration(self, total_people: int, max_zone: int) -> float:
        """
        Calculates the ratio of people clustered in the single busiest zone.
        """
        if total_people <= 0:
            return 0.0
        return min(max_zone / total_people, 1.0)

    def _calculate_congestion_score(self, density: float, concentration: float) -> int:
        """
        Mathematical Formula:
        Congestion Score = (Density * Density_Weight) + (Concentration * Hotspot_Weight)
        Mapped to a 0-100 scale.
        """
        # Clamp inputs
        density = max(0.0, min(density, 1.0))
        concentration = max(0.0, min(concentration, 1.0))
        
        raw_score = (density * self.density_weight) + (concentration * self.hotspot_weight)
        score_100 = int(round(raw_score * 100))
        
        # Ensure it stays within bounds
        return max(0, min(score_100, 100))

    def _determine_risk_level(self, congestion_score: int) -> RiskLevel:
        """
        Rule-based mapping from numeric score to Risk Level.
        """
        if congestion_score < 30:
            return RiskLevel.LOW
        elif congestion_score < 60:
            return RiskLevel.MODERATE
        elif congestion_score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def analyze_risk(self, data: RiskInput) -> RiskOutput:
        """
        Executes the risk analysis algorithm.
        """
        concentration = self._calculate_hotspot_concentration(data.people_count, data.max_zone_count)
        congestion_score = self._calculate_congestion_score(data.density_score, concentration)
        risk_level = self._determine_risk_level(congestion_score)
        
        logger.info(f"Risk Analysis - Score: {congestion_score}, Level: {risk_level.value}")
        
        return RiskOutput(
            risk_level=risk_level,
            congestion_score=congestion_score,
            hotspot_concentration=round(concentration, 4)
        )

# --- Example Usage ---
if __name__ == "__main__":
    agent = RiskAnalysisAgent()
    test_input = RiskInput(people_count=1200, density_score=0.81, max_zone_count=630)
    output = agent.analyze_risk(test_input)
    print(output.model_dump_json(indent=2))
