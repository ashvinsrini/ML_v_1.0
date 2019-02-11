from django import forms
class FormName(forms.Form):
    name = forms.CharField()
    email = forms.EmailField()
    filepath = forms.CharField(widget = forms.Textarea)
    label = forms.CharField(widget = forms.Textarea)
    ColumnsConsidered = forms.CharField(widget = forms.Textarea)
    #Imputation =  forms.CharField(widget = forms.Textarea)
    #Target =  forms.CharField(widget = forms.Textarea)
    #Classifier = forms.CharField(widget = forms.Textarea)
    #ModelingType = forms.CharField(widget = forms.Textarea)
    separator = forms.CharField(widget = forms.Textarea)


def clean(self):
    all_clean_data = super().clean()
    lab = all_clean_data['label']
    label_list = ['supervised','unsupervised']
    if lab not in label_list:
        raise forms.ValidationError("Make sure the imputation is entered from the list")
