from django.shortcuts import render,HttpResponse
import joblib
# Create your views here.

def index(request):
    return render(request,'home.html',{})

def result(request):

    modell=joblib.load('final_model.sav')
    lis=[]

    lis.append(request.GET['age'])
    lis.append(request.GET['sex'])
    lis.append(request.GET['cp'])
    lis.append(request.GET['trestbps'])
    lis.append(request.GET['chol'])
    lis.append(request.GET['fbs'])
    lis.append(request.GET['restecg'])
    lis.append(request.GET['thalach'])
    lis.append(request.GET['exang'])
    lis.append(request.GET['oldpeak'])
    lis.append(request.GET['slope'])
    lis.append(request.GET['ca'])
    lis.append(request.GET['thal'])  

    print(lis)

    ans=modell.predict([lis])
    print(ans)

    return render(request,'result.html',{'ans':ans})
