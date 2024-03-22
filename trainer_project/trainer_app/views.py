import datetime
import secrets
from django.http import HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from statistics import mean
import json
from django.forms.models import model_to_dict
from .file_processing import *
from .forms import *
from .mail import *
from .models import *
from .recommendations import *


@csrf_exempt
def register(request):
    """
    Registers user, checks if email is valid and not already registered
    @param request: POST request with email and password
    @return: token if registration is successful, error message otherwise
    """
    query_dict = request.POST.copy()
    if len(query_dict["password"]) < 4:
        return HttpResponse("password_error", headers={
            "status": f"Ensure this value has at least 4 characters (it has {len(query_dict['password'])})."})
    person_id = Authentication.objects.all().count() + 1
    query_dict.appendlist("id", person_id)
    form = AuthenticationForm(query_dict)
    if form.is_valid():
        token = secrets.token_hex(16)
        user = form.save(commit=False)
        user.token = token
        user.save()
        return HttpResponse(token, headers={"status": "User is successfully registered."})
    else:
        if "email" in form.errors.as_data():
            return HttpResponse("email_error", headers={"status": str(form.errors["email"][0])})
        if "password" in form.errors.as_data():
            return HttpResponse("password_error", headers={"status": str(form.errors["password"][0])})
        return HttpResponse("unknown_error", headers={"status": "Server error. Please contact support."})


@csrf_exempt
def login(request):
    """
    Logins user, checks email and password
    @param request: POST request with email and password
    @return: token if registration is successful, error message otherwise
    """
    email = request.POST.get("email")
    user = Authentication.objects.filter(email=email)
    if not user.exists():
        return HttpResponse("email_error", headers={"status": "Email is not registered."})
    user = user[0]
    if user.password != request.POST.get("password"):
        return HttpResponse("password_error", headers={"status": "Incorrect password."})
    token = secrets.token_hex(16)
    user.token = token
    user.save()
    return HttpResponse(token, headers={"status": "User is successfully logged in."})


@csrf_exempt
def logout(request):
    """
    Logouts user (deletes token)
    @param request: POST request with user's token
    @return: success message if logout is successful, error message otherwise
    """
    token = request.POST.get("token")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    user = user[0]
    user.token = None
    user.save()
    return HttpResponse("ok", headers={"status": "User is successfully logged out."})


@csrf_exempt
def password_recovery(request):
    """
    Request for password recovery
    @param request: POST request with user's email
    @return: four-number code for email confirmation if email is registered, error message otherwise
    """
    email = request.POST.get("email")
    user = Authentication.objects.filter(email=email)
    if not user.exists():
        return HttpResponse("email_error", headers={"status": "Email is not registered."})
    code = secrets.choice(seq=range(1000, 10000))
    send_message(email, code)
    return HttpResponse(code, headers={"status": "Email is registered."})


@csrf_exempt
def password_update(request):
    """
    Updates password
    @param request: POST request with user's new password
    @return: token if password update is successful, error message otherwise
    """
    email = request.POST.get("email")
    user = Authentication.objects.filter(email=email)
    if not user.exists():
        return HttpResponse("email_not_found_error", headers={"status": "Server error. Please contact support."})
    user = user[0]
    user.password = request.POST.get("password")
    token = secrets.token_hex(16)
    user.token = token
    user.save()
    return HttpResponse(token, headers={"status": "Password is successfully changed."})


@csrf_exempt
def upload_file(request):
    """
    Uploads user's file and analyses it
    @param request: POST request with user's token, file and chosen analysis parameters
    @return: file id if analysis is successful, error message otherwise
    """
    token = request.POST.get("token")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    user_id = user[0].id
    query_dict = request.POST.copy()
    query_dict.appendlist("timestamp", datetime.datetime.now())
    query_dict.appendlist("user_id", user_id)
    file_id = FileInfo.objects.all().count() + 1
    query_dict.appendlist("id", file_id)
    query_dict.pop("token")
    form = FileInfoForm(query_dict, request.FILES)
    if form.is_valid():
        form.save()
        file = FileInfo.objects.get(id=file_id)
        get_screenshot(file)
        if not file_processing(file):
            return HttpResponse("analysis_error", headers={"status": "Server error. Please contact support."})
        return HttpResponse(str(file_id), headers={"status": "File is successfully uploaded."})
    elif "filename" in form.errors.as_data():
        return HttpResponse("filename_error", headers={"status": str(form.errors["filename"][0])})
    else:
        return HttpResponse("parsing_error", headers={"status": "Server error. Please contact support."})


@csrf_exempt
def archive_number_of_files(request):
    """
    Sends number of user's files and their ids
    @param request: POST request with user's token
    @return: files' count and list of their ids in JSON format if request is successful, error message otherwise
    """
    token = request.POST.get("token")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    user_id = user[0].id
    files = FileInfo.objects.filter(user_id=user_id)
    file_ids = list()
    for file in files:
        file_ids.append(file.id)
    files_info = {"num_of_files": files.count(), "file_ids": file_ids}
    return HttpResponse(json.dumps(files_info), headers={"status": "User's files are successfully found."})


@csrf_exempt
def archive_file_info(request):
    """
    Sends base file info
    @param request: POST request with user's token and file id
    @return: file's name and date of receiving in JSON format if request is successful, error message otherwise
    """
    token = request.POST.get("token")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    file = FileInfo.objects.filter(id=request.POST.get("file_id"))
    if not file.exists():
        return HttpResponse("file_not_found_error", headers={"status": "Server error. Please contact support."})
    file = file[0]
    if user[0].id != file.user_id.id:
        return HttpResponse("file_and_token_do_not_match_error",
                            headers={"status": "Server error. Please contact support."})
    file_info = {"filename": file.filename, "datetime": file.timestamp.strftime("%Y:%m:%d %H:%M:%S")}
    return HttpResponse(json.dumps(file_info), headers={"status": "File is successfully found."})


@csrf_exempt
def archive_file_image(request):
    """
    Sends file image (in png format) for archive preview
    @param request: POST request with user's token and file id
    @return: image file (screenshot from video) if request is successful, error message otherwise
    """
    token = request.POST.get("token")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    file = FileInfo.objects.filter(id=request.POST.get("file_id"))
    if not file.exists():
        return HttpResponse("file_not_found_error", headers={"status": "Server error. Please contact support."})
    file = file[0]
    if user[0].id != file.user_id.id:
        return HttpResponse("file_and_token_do_not_match_error",
                            headers={"status": "Server error. Please contact support."})
    path = file.file.path
    image_path = path[:path.rfind('.')] + '.png'
    return FileResponse(open(image_path, "rb"), headers={"status": "File is successfully found."})


@csrf_exempt
def video_file(request):
    """
    Sends video file
    @param request: POST request with user's token and file id
    @return: video file (same format as was sent) if request is successful, error message otherwise
    """
    token = request.POST.get("token")
    file_id = request.POST.get("file_id")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    file = FileInfo.objects.filter(id=file_id)
    if not file.exists():
        return HttpResponse("file_not_found_error", headers={"status": "Server error. Please contact support."})
    if user[0].id != file[0].user_id.id:
        return HttpResponse("file_and_token_do_not_match_error",
                            headers={"status": "Server error. Please contact support."})
    path = file[0].file.path
    return FileResponse(open(path, "rb"), headers={"status": "File is successfully found."})


@csrf_exempt
def file_statistics(request):
    """
    Sends file analysis results
    @param request: POST request with user's token and file id
    @return: file statistics converted to text format, timestamps and filler words in JSON format
    if request is successful, error message otherwise
    """
    token = request.POST.get("token")
    file_id = request.POST.get("file_id")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    file = FileInfo.objects.filter(id=file_id)
    if not file.exists():
        return HttpResponse("file_not_found_error", headers={"status": "Server error. Please contact support."})
    if user[0].id != file[0].user_id.id:
        return HttpResponse("file_and_token_do_not_match_error",
                            headers={"status": "Server error. Please contact support."})
    file = FileAnalysis.objects.get(file_id=file_id)
    # Transform real numbers to text grades according to limit values
    data = model_to_dict(file)
    data.pop("text")
    data.pop("id")
    data.pop("file_id")
    data = convert_to_text(data)

    # Save timestamps lists in string format
    background_noise_timestamps, low_speech_rate_timestamps = [], []
    timestamps = FileTimestamps.objects.filter(file_id=file_id)
    for timestamp in timestamps:
        start = timestamp.start.strftime("%H:%M:%S")
        end = timestamp.end.strftime("%H:%M:%S")
        if timestamp.time_period_type == 0:
            background_noise_timestamps.append([start, end])
        else:
            low_speech_rate_timestamps.append([start, end])
    data["background_noise_timestamps"] = background_noise_timestamps
    data["low_speech_rate_timestamps"] = low_speech_rate_timestamps
    filler_words_lst = FillerWords.objects.filter(file_id=file_id, most_common=True)
    filler_words = [word.word_or_phrase for word in filler_words_lst]
    data["filler_words"] = filler_words
    res = json.dumps(data, default=float)
    return HttpResponse(res)


@csrf_exempt
def statistics(request):
    """
    Sends user statistics (numbers for each file, filler words count)
    @param request: POST request with user's token
    @return: user statistics as lists of values, filler words and their percentage in JSON format
    if request is successful, error message otherwise
    """
    token = request.POST.get("token")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    all_data, ids_lst = get_user_stats(user[0].id)
    data = dict()
    demo_vals = ["speech_rate", "angle"]
    for val in demo_vals:
        data[val] = all_data[val]

    filler_words_objects = FillerWords.objects.filter(file_id__in=ids_lst)
    filler_words_dict = dict()
    for filler_word_object in filler_words_objects:
        word = filler_word_object.word_or_phrase
        if word in FillerWordsAndPhrases.params["parasites"]:
            if word not in filler_words_dict:
                filler_words_dict[word] = 0
            filler_words_dict[word] += filler_word_object.occurrence
    total_count = sum(list(filler_words_dict.values()))
    filler_words_list = sorted(list(filler_words_dict.items()), key=lambda x: -x[1])
    if len(filler_words_list) > 5:
        filler_words_list = filler_words_list[:4]
        other_count = total_count - sum([filler_word[1] for filler_word in filler_words_list])
        filler_words_list.append(("Прочее", other_count))
    data["filler_words"] = [filler_word[0] for filler_word in filler_words_list]
    data["filler_words_percentage"] = [filler_word[1] / total_count for filler_word in filler_words_list]
    res = json.dumps(data, default=float)
    return HttpResponse(res)


@csrf_exempt
def recommendations(request):
    """
    Sends recommendations for each parameter based on averaged files statistics
    @param request: POST request with user's token
    @return: text recommendations in JSON format if request is successful, error message otherwise
    """
    token = request.POST.get("token")
    user = Authentication.objects.filter(token=token)
    if not user.exists():
        return HttpResponse("token_not_found_error", headers={"status": "Server error. Please contact support."})
    data, _ = get_user_stats(user[0].id)
    for key in data:
        if len(data[key]) == 0:
            data[key].append(-1)
        new_val = mean(data[key])
        constants_key = key
        if new_val > CONSTANTS[constants_key][1]:
            new_val = 2
        elif new_val > CONSTANTS[constants_key][0]:
            new_val = 1
        elif new_val > -1:
            new_val = 0
        data[key] = recs[key][new_val]

    res = json.dumps(data, default=float)
    return HttpResponse(res)


def file_processing(file: FileInfo):
    """
    Analyses file according to chosen parameters and saves results
    @param file: video file instance
    @return: True if analysis is successful, False if errors occurred
    """
    try:
        file_process = FileProcessingSystem(file)
        if file.emotionality:
            file_process.get_emotionality()
        if file.gestures or file.clothes or file.angle or file.glances:
            if file.gestures:
                file_process.get_gestures()
            if file.clothes:
                file_process.get_clothes()
            if file.angle:
                file_process.get_incorrect_angle()
            if file.glances:
                file_process.get_incorrect_glances()
        if file.clean_speech or file.speech_rate or file.background_noise or file.intelligibility:
            file_process.get_transcription()
            if file.background_noise:
                file_process.get_background_noise()
            if file.speech_rate:
                file_process.get_speech_rate()
            if file.clean_speech:
                file_process.get_filler_words()
            if file.intelligibility:
                file_process.get_intelligibility()
    except Exception as e:
        return False
    return True


def get_screenshot(file: FileInfo):
    """
    Creates and saves file image (video screenshot)
    @param file: video file instance
    """
    path = file.file.path
    clip = mp_editor.VideoFileClip(path)
    image_path = path[:path.rfind('.')] + '.png'
    clip.save_frame(image_path, t=0)


def convert_to_text(data):
    """
    Converts real values to text
    @param data: dict with values for all analysis parameters
    @return: dict with text evaluation for each parameter
    """
    new_data = dict()
    for key in data:
        val = data[key]
        if val < 0:
            continue
        constants_key = key
        new_val = "No info"
        if val > CONSTANTS[constants_key][1]:
            new_val = "Высокая"
        elif val > CONSTANTS[constants_key][0]:
            new_val = "Средняя"
        elif val > -1:
            new_val = "Низкая"
        if key == "clothes":
            new_val = "Подходящий" if val > CONSTANTS[constants_key][0] else "Неподходящий"
        new_data[key] = new_val
    return new_data


def get_user_stats(user_id, get_unknown=False):
    """
    Aggregates values from all user's video files
    @param user_id: user's id to search files
    @param get_unknown: whether to collect default values (-1)
    @return: lists with real values - aggregated files' scores, and list with all user's files ids
    """
    files = FileInfo.objects.filter(user_id=user_id)
    ids_lst = [file.id for file in files]
    files = FileAnalysis.objects.filter(file_id__in=ids_lst)
    ignore_fields = ["id", "file_id", "text"]
    field_names = [field.name for field in FileAnalysis._meta.get_fields() if field.name not in ignore_fields]
    data = {field_name: [] for field_name in field_names}

    for file in files:
        file_dict = model_to_dict(file)
        for key in file_dict.keys():
            if key not in ignore_fields:
                if get_unknown or (not get_unknown and file_dict[key] > -0.5):
                    data[key].append(file_dict[key])

    return data, ids_lst
