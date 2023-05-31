from django.shortcuts import render
from joblib import load

# Create your views here.
def home(request):
    return render(request, 'main.html')

model = load('./Models/Malaria-diagnosis.joblib')
# Create your views here.
def predictor(request):
    if request.method == 'POST':
        temperature = request.POST['temperature']
        parasite_density = request.POST['parasite_density']
        wbc_count = request.POST['wbc_count']
        hb_level = request.POST['hb_level']

        hematocrit = request.POST['hematocrit']
        mean_cell_volume = request.POST['mean_cell_volume']
        mean_corp_hb = request.POST['mean_corp_hb']
        mean_cell_hb_conc = request.POST['mean_cell_hb_conc']

        platelet_count = request.POST['platelet_count']
        platelet_distr_width = request.POST['platelet_distr_width']
        mean_platelet_vl = request.POST['mean_platelet_vl']
        neutrophils_percent = request.POST['neutrophils_percent']

        lymphocytes_percent = request.POST['lymphocytes_percent']
        mixed_cells_percent = request.POST['mixed_cells_percent']
        neutrophils_count = request.POST['neutrophils_count']
        lymphocytes_count = request.POST['lymphocytes_count']
        mixed_cells_count = request.POST['mixed_cells_count']



        y_pred = model.predict([[temperature, parasite_density, wbc_count, hb_level,
                                hematocrit, mean_cell_volume, mean_corp_hb, mean_cell_hb_conc,
                                platelet_count, platelet_distr_width, mean_platelet_vl,
                                neutrophils_percent, lymphocytes_percent, mixed_cells_percent,
                                neutrophils_count, lymphocytes_count, mixed_cells_count]])
        #print(y_pred)
        #if y_pred[0] == 0
        #return render(request, 'main.html', {'result' : y_pred})
        return render(request, 'main.html', {'result': str(y_pred[0]).strip("[]'")})
    return render(request, 'main.html')

    
