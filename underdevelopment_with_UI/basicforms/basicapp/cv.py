from django import forms
class FormName(forms.Form):
    name = forms.CharField()
    email = forms.EmailField()
    batch_size = forms.CharField()
    num_classes = forms.CharField()
    Epochs = forms.CharField()
    optimizer = forms.CharField()
    data_augmentation = forms.CharField()
    demo = forms.CharField()
    filepath = forms.CharField()
    target = forms.CharField()
    imght = forms.CharField()
    imgwdt = forms.CharField()
    rgb = forms.CharField()


def clean(self):
    all_clean_data = super().clean()
    lab = all_clean_data['label']
    label_list = ['supervised','unsupervised']
    if lab not in label_list:
        raise forms.ValidationError("Make sure the imputation is entered from the list")
