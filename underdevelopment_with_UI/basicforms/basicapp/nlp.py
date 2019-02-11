from django import forms
class FormName(forms.Form):
    name = forms.CharField()
    email = forms.EmailField()
    label = forms.CharField(widget = forms.Textarea)
    paths = forms.CharField(widget = forms.Textarea)
    numOfCat = forms.CharField()
    catType = forms.CharField(widget = forms.Textarea)
    train = forms.CharField(widget = forms.Textarea)


def clean(self):
    all_clean_data = super().clean()
    lab = all_clean_data['label']
    label_list = ['supervised','unsupervised']
    if lab not in label_list:
        raise forms.ValidationError("Make sure the imputation is entered from the list")
