from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants.constant import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import VehicleData, VehicleDataClassifier

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    """

    def __init__(self, request: Request):
        self.request: Request = request
        self.driving_experience: Optional[int] = None
        self.education: Optional[int] = None
        self.income: Optional[int] = None
        self.vehicle_year_before_2015: Optional[int] = None
        self.credit_score: Optional[int] = None
        self.annual_mileage: Optional[float] = None
        self.age: Optional[int] = None
        self.gender: Optional[int] = None
        self.vehicle_ownership: Optional[int] = None
        self.married: Optional[int] = None
        self.children: Optional[int] = None
        self.speeding_violations: Optional[int] = None
        self.past_accidents: Optional[int] = None

    async def get_vehicle_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.driving_experience = form.get("driving_experience")
        self.education = form.get("education")
        self.income = form.get("income")
        self.vehicle_year_before_2015 = form.get("vehicle_year")
        self.credit_score = form.get("credit_score")
        self.annual_mileage = form.get("annual_mileage")
        self.age = form.get("age")
        self.gender = form.get("gender")
        self.vehicle_ownership = form.get("vehicle_ownership")
        self.married = form.get("married")
        self.children = form.get("children")
        self.speeding_violations = form.get("speeding_violations")
        self.past_accidents = form.get("past_accidents")


# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
        "vehicledata.html",
        {"request": request, "context": "Rendering"},
    )


# Route to handle form submission and make predictions
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        vehicle_data = VehicleData(
            driving_experience=form.driving_experience,
            education=form.education,
            income=form.income,
            vehicle_year_before_2015=form.vehicle_year_before_2015,
            credit_score=form.credit_score,
            annual_mileage=form.annual_mileage,
            age=form.age,
            gender=form.gender,
            vehicle_ownership=form.vehicle_ownership,
            married=form.married,
            children=form.children,
            speeding_violations=form.speeding_violations,
            past_accidents=form.past_accidents,
        )

        # Convert form data into a DataFrame for the model
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = VehicleDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=vehicle_df)[0]

        # Interpret the prediction result as 'Response-Yes' or 'Response-No'
        status = "Response-Claim" if value == 1 else "Response-No Claim"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "vehicledata.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}


# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
