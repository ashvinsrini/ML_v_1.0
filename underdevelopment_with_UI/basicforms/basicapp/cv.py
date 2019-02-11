from django import forms
class FormName(forms.Form):
    name = forms.CharField()
    email = forms.EmailField()
    batch_size = forms.CharField(widget = forms.Textarea)
    num_classes = forms.CharField(widget = forms.Textarea)
    Epochs = forms.CharField(widget = forms.Textarea)
    optimizer = forms.CharField(widget = forms.Textarea)
    data_augmentation = forms.CharField(widget = forms.Textarea)
    demo = forms.CharField(widget = forms.Textarea)



def clean(self):
    all_clean_data = super().clean()
    lab = all_clean_data['label']
    label_list = ['supervised','unsupervised']
    if lab not in label_list:
        raise forms.ValidationError("Make sure the imputation is entered from the list")
