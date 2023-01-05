import numpy as np
from sklearn.linear_model import LinearRegression
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

# قراءة البيانات وتدريب النموذج
data = np.genfromtxt('Data.csv', delimiter=',')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]
model = LinearRegression().fit(X, y)

class SalaryPredictionApp(App):
    def build(self):
        # إنشاء العنصر الرئيسي للواجهة الرسومية
        layout = BoxLayout(orientation='vertical')

        # إنشاء مربع النص للإدخال
        self.years_input = TextInput(hint_text="Enter years of experience")
        layout.add_widget(self.years_input)

        # إنشاء الزر الذي يطلب من المستخدم الإدخال
        self.predict_button = Button(text="Predict", on_press=self.predict_salary)
        layout.add_widget(self.predict_button)

        # إنشاء العلامة التي تعرض النتيجة
        self.result_label = Label(text="")
        layout.add_widget(self.result_label)

        return layout

    def predict_salary(self, instance):
        years = float(self.years_input.text)
        predicted_salary = model.predict([[years]])
        self.result_label.text = f'Predicted salary for {years} years of experience: {predicted_salary}'

if __name__ == '__main__':
    SalaryPredictionApp().run()
