# Patient UI view
def patient_ui(request):
    return render(request, 'patient/patient_ui.html')
# Signin page view
def signin_page(request):
    return render(request, 'signin_page/index.html')
# Placeholder views for missing endpoints
from django.http import HttpResponse

def admin_ui(request):
    return HttpResponse('admin_ui placeholder')


def pviewprofile(request, patientusername):
    return HttpResponse(f'pviewprofile placeholder for {patientusername}')

def pconsultation_history(request):
    return HttpResponse('pconsultation_history placeholder')

def consult_a_doctor(request):
    return HttpResponse('consult_a_doctor placeholder')

def make_consultation(request, doctorusername):
    return HttpResponse(f'make_consultation placeholder for {doctorusername}')

def rate_review(request, consultation_id):
    return HttpResponse(f'rate_review placeholder for {consultation_id}')

def dconsultation_history(request):
    return HttpResponse('dconsultation_history placeholder')

def dviewprofile(request, doctorusername):
    return HttpResponse(f'dviewprofile placeholder for {doctorusername}')

def doctor_ui(request):
    return HttpResponse('doctor_ui placeholder')

def consultationview(request, consultation_id):
    return HttpResponse(f'consultationview placeholder for {consultation_id}')

def close_consultation(request, consultation_id):
    return HttpResponse(f'close_consultation placeholder for {consultation_id}')

def post(request):
    return HttpResponse('post placeholder')

def chat_messages(request):
    return HttpResponse('chat_messages placeholder')
# updated views.py (only the checkdisease function)

from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.models import User
from .models import diseaseinfo
import os
import joblib
import numpy as np

# Load model and encoder
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_model', 'model.pkl'))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trained_model', 'label_encoder.pkl'))
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Symptoms used in the training data
symptomslist = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination',
    'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
    'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating',
    'dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes',
    'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze'
]

alphabaticsymptomslist = sorted(symptomslist)

def checkdisease(request):
    if request.method == 'GET':
        return render(request, 'patient/checkdisease/checkdisease.html', {"list2": alphabaticsymptomslist})

    elif request.method == 'POST':
        try:
            inputno = int(request.POST.get("noofsym", 0))
            psymptoms = request.POST.getlist("symptoms[]")

            if inputno == 0 or not psymptoms:
                return JsonResponse({'predicteddisease': "none", 'confidencescore': 0})

            # Create binary vector for symptoms
            testingsymptoms = [1 if symptom in psymptoms else 0 for symptom in symptomslist]
            inputtest = [testingsymptoms]

            predicted = model.predict(inputtest)
            y_pred_proba = model.predict_proba(inputtest)
            confidence = format(np.max(y_pred_proba) * 100, '.0f')
            predicted_disease = label_encoder.inverse_transform(predicted)[0]

            # Optional: add your custom doctor mapping here
            consultdoctor = "General Physician"

            # Save to DB
            patientusername = request.session.get('patientusername')
            puser = User.objects.get(username=patientusername)

            diseaseinfo_new = diseaseinfo(
                patient=puser.patient,
                diseasename=predicted_disease,
                no_of_symp=inputno,
                symptomsname=psymptoms,
                confidence=confidence,
                consultdoctor=consultdoctor
            )
            diseaseinfo_new.save()
            request.session['diseaseinfo_id'] = diseaseinfo_new.id
            request.session['doctortype'] = consultdoctor

            return JsonResponse({
                'predicteddisease': predicted_disease,
                'confidencescore': confidence,
                'consultdoctor': consultdoctor
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'predicteddisease': "error",
                'confidencescore': 0,
                'errormsg': str(e)
            })

# Home view must be outside of all other functions
def home(request):
    return render(request, 'homepage/home.html')
