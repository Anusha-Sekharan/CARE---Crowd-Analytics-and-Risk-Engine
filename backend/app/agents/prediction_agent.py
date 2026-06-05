import logging
from pydantic import BaseModel, Field
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# --- Schemas ---

class PredictionInput(BaseModel):
    current_people_count: int = Field(..., description="Total people count")
    congestion_score: int = Field(..., description="Congestion score (0-100) from Agent 3")
    hotspot_concentration: float = Field(..., description="Concentration ratio from Agent 3")
    time_horizon_mins: int = Field(15, description="Minutes into the future to predict")

class PredictionOutput(BaseModel):
    future_people_count: int = Field(..., description="Estimated people count in the future")
    future_congestion_score: int = Field(..., description="Estimated future congestion score")
    stampede_risk: int = Field(..., description="Calculated probability of a stampede or crush event (0-100)")
    prediction_method: str = Field(..., description="Method used for prediction (V1: Rule-based)")

# --- Agent Implementation ---

class PredictionAgent:
    """
    Agent 4: Prediction Agent
    V1: Uses a mathematical rule-based model to simulate crowd growth and future risk.
    V2 (Planned): Will utilize LSTM/Transformers for timeseries forecasting.
    """
    
    def __init__(self, base_growth_rate: float = 0.05):
        """
        Initializes the prediction agent.
        
        Args:
            base_growth_rate: The assumed baseline rate of crowd growth per 15 minutes.
        """
        self.base_growth_rate = base_growth_rate

    def _simulate_crowd_growth(self, current_count: int, congestion: int, horizon_mins: int) -> int:
        """
        Mathematical Formula for V1 Crowd Growth:
        Future_Count = Current_Count * (1 + (Base_Rate * Modifier * (Horizon / 15)))
        
        Modifier is based on congestion. If congestion is extremely high, 
        growth slows down due to physical space limitations (saturation).
        """
        # If congestion is > 90, physical capacity is reached, growth halts.
        if congestion >= 90:
            growth_modifier = 0.0
        else:
            # Growth accelerates up to 70% congestion, then decelerates as space fills
            growth_modifier = 1.0 - (max(0, congestion - 70) / 20.0)
            
        rate_over_time = self.base_growth_rate * growth_modifier * (horizon_mins / 15.0)
        future_count = int(current_count * (1.0 + rate_over_time))
        
        return future_count

    def _estimate_stampede_risk(self, congestion: int, concentration: float, future_growth_ratio: float) -> int:
        """
        Mathematical Formula for Stampede Risk:
        Stampede Risk = (Congestion * 0.5) + (Concentration * 100 * 0.3) + (Growth_Momentum * 0.2)
        
        High concentration in a single spot + High overall congestion + Positive growth = Disaster Risk
        """
        momentum_score = min(100, max(0, (future_growth_ratio - 1.0) * 100 * 5)) # E.g., 5% growth -> 25 score
        
        raw_risk = (congestion * 0.5) + (concentration * 100 * 0.3) + (momentum_score * 0.2)
        risk_100 = int(round(raw_risk))
        
        return max(0, min(risk_100, 100))

    def predict_future_state(self, data: PredictionInput) -> PredictionOutput:
        """
        Executes the rule-based prediction algorithm.
        """
        # 1. Predict future count
        future_count = self._simulate_crowd_growth(
            current_count=data.current_people_count,
            congestion=data.congestion_score,
            horizon_mins=data.time_horizon_mins
        )
        
        growth_ratio = future_count / max(1, data.current_people_count)
        
        # 2. Predict future congestion (linear scaling with count for V1)
        future_congestion = min(100, int(data.congestion_score * growth_ratio))
        
        # 3. Predict stampede risk
        stampede_risk = self._estimate_stampede_risk(
            congestion=future_congestion,
            concentration=data.hotspot_concentration,
            future_growth_ratio=growth_ratio
        )
        
        logger.info(f"Prediction - Future Count: {future_count}, Stampede Risk: {stampede_risk}")
        
        return PredictionOutput(
            future_people_count=future_count,
            future_congestion_score=future_congestion,
            stampede_risk=stampede_risk,
            prediction_method="V1_RULE_BASED_SIMULATION"
        )

# --- Example Usage ---
if __name__ == "__main__":
    agent = PredictionAgent()
    # Using inputs derived from previous agents
    test_input = PredictionInput(
        current_people_count=1200, 
        congestion_score=84, 
        hotspot_concentration=0.525, # 630 / 1200
        time_horizon_mins=15
    )
    output = agent.predict_future_state(test_input)
    print(output.model_dump_json(indent=2))
