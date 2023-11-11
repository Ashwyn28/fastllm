import pytest
from app.main import PlannerMessages

class TestPlannerMessages:
    def __init__(self) -> None:
        PlannerMessages()

    def test_make_activity(self):

        activity_str = "make coffee"
        time_str = "10/12/2023 09:00:00"
        
        PlannerMessages.make_activity(time_str, activity_str)

        assert PlannerMessages.get_messages() 
