import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/USER/Desktop/ΓΙΑΝΝΗΣ/Msc BIG DATA ANALYTICS/THESIS/df_new.csv", encoding = "latin1", delimiter=';')

days_of_week = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
descent_code_to_name = {
    'A': 'Other Asian', 'B': 'Black', 'C': 'Chinese', 'D': 'Cambodian', 'F': 'Filipino', 
    'G': 'Guamanian', 'H': 'Hispanic/Latin/Mexican', 'I': 'American Indian/Alaskan Native', 
    'J': 'Japanese', 'K': 'Korean', 'L': 'Laotian', 'O': 'Other', 'P': 'Pacific Islander', 
    'S': 'Samoan', 'U': 'Hawaiian', 'V': 'Vietnamese', 'W': 'White', 'X': 'Unknown', 'Z': 'Asian Indian'
}
sex_code_to_name = {
    'M': 'Male', 'F': 'Female'
}

# Συνάρτηση για να αντικαταστήσουμε τα αρχικά γράμματα με τις πλήρεις λέξεις
def replace_descent_codes(series):
    return series.map(descent_code_to_name)

def replace_sex_codes(series):
    return series.map(sex_code_to_name)

# Συνάρτηση για να υπολογίσουμε την πιθανότητα και να βρούμε τη μεγαλύτερη ανά ημέρα της εβδομάδας
def get_max_prob_by_day(df, day_of_week):
    filtered_df = df[df['Day of Week'] == day_of_week]
    if filtered_df.empty:
        return None
    filtered_df['Victim Descent'] = replace_descent_codes(filtered_df['Victim Descent'])
    filtered_df['Victim Sex'] = replace_sex_codes(filtered_df['Victim Sex'])
    result = {}
    columns_to_analyze = ['Age Category', 'Victim Sex', 'Victim Descent', 'Area Name', 'Crime Severity', 'Crime Category']
    for col in columns_to_analyze:
        value_counts = filtered_df[col].value_counts(normalize=True)
        max_value = value_counts.idxmax()
        max_prob = value_counts.max()
        result[col] = (value_counts, max_value, max_prob)
    return result

# Συνάρτηση για να υπολογίσουμε την πιθανότητα και να βρούμε τη μεγαλύτερη ανά μήνα
def get_max_prob_by_month(df, month):
    filtered_df = df[df['Month'] == month]
    if filtered_df.empty:
        return None
    filtered_df['Victim Descent'] = replace_descent_codes(filtered_df['Victim Descent'])
    filtered_df['Victim Sex'] = replace_sex_codes(filtered_df['Victim Sex'])
    result = {}
    columns_to_analyze = ['Age Category', 'Victim Sex', 'Victim Descent', 'Area Name', 'Crime Severity', 'Crime Category']
    for col in columns_to_analyze:
        value_counts = filtered_df[col].value_counts(normalize=True)
        max_value = value_counts.idxmax()
        max_prob = value_counts.max()
        result[col] = (value_counts, max_value, max_prob)
    return result

# Συνάρτηση για να υπολογίσουμε την πιθανότητα και να βρούμε τη μεγαλύτερη ανά περιοχή
def get_max_prob_by_area(df, area_name):
    filtered_df = df[df['Area Name'] == area_name]
    if filtered_df.empty:
        return None
    filtered_df['Victim Descent'] = replace_descent_codes(filtered_df['Victim Descent'])
    filtered_df['Victim Sex'] = replace_sex_codes(filtered_df['Victim Sex'])
    result = {}
    columns_to_analyze = ['Age Category', 'Victim Sex', 'Victim Descent', 'Month', 'Crime Severity', 'Crime Category']
    for col in columns_to_analyze:
        value_counts = filtered_df[col].value_counts(normalize=True)
        max_value = value_counts.idxmax()
        max_prob = value_counts.max()
        result[col] = (value_counts, max_value, max_prob)
    return result

# Συνάρτηση για να δημιουργήσουμε τα γραφήματα
def plot_probabilities(value_counts, title, chart_type='bar'):
    fig, ax = plt.subplots()
    colors = plt.cm.Paired(range(len(value_counts)))
    if chart_type == 'pie':
        value_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax)
        ax.set_ylabel('')
    else:
        value_counts.plot(kind='bar', color=colors, ax=ax)
        ax.set_ylabel('Probability')
    ax.set_title(title)
    st.pyplot(fig)

# Δημιουργία του Streamlit app
st.title('Crime Probability Dashboard')

# Επιλογή ανάλυσης
analysis_type = st.radio('Select Analysis Type', ['Day of Week', 'Month', 'Area'])

if analysis_type == 'Day of Week':
    # Επιλογή μέρας της εβδομάδας
    day_of_week_number = st.selectbox('Select Day of Week', list(days_of_week.keys()), format_func=lambda x: days_of_week[x])
    
    # Υπολογισμός των πιθανοτήτων
    max_probabilities = get_max_prob_by_day(df, day_of_week_number)
    
    # Εμφάνιση των αποτελεσμάτων
    if max_probabilities:
        st.subheader(f'Max Probabilities for {days_of_week[day_of_week_number]}')
        for key, (value_counts, max_value, max_prob) in max_probabilities.items():
            if key == 'Area Name':
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            elif key == 'Victim Descent':
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            elif key == 'Victim Sex':
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            else:
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            if key in ['Victim Sex', 'Crime Severity']:
                plot_probabilities(value_counts, f'{key} Distribution for {days_of_week[day_of_week_number]}', chart_type='pie')
            else:
                plot_probabilities(value_counts, f'{key} Distribution for {days_of_week[day_of_week_number]}')
    else:
        st.write(f"No data available for {days_of_week[day_of_week_number]}")

elif analysis_type == 'Month':
    # Επιλογή μήνα
    month_number = st.selectbox('Select Month', list(months.keys()), format_func=lambda x: months[x])
    
    # Υπολογισμός των πιθανοτήτων
    max_probabilities = get_max_prob_by_month(df, month_number)
    
    # Εμφάνιση των αποτελεσμάτων
    if max_probabilities:
        st.subheader(f'Max Probabilities for {months[month_number]}')
        for key, (value_counts, max_value, max_prob) in max_probabilities.items():
            if key == 'Area Name':
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            elif key == 'Victim Descent':
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            elif key == 'Victim Sex':
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            else:
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            if key in ['Victim Sex', 'Crime Severity']:
                plot_probabilities(value_counts, f'{key} Distribution for {months[month_number]}', chart_type='pie')
            else:
                plot_probabilities(value_counts, f'{key} Distribution for {months[month_number]}')
    else:
        st.write(f"No data available for {months[month_number]}")

elif analysis_type == 'Area':
    # Επιλογή περιοχής
    area_name = st.selectbox('Select Area Name', df['Area Name'].unique())
    
    # Υπολογισμός των πιθανοτήτων
    max_probabilities = get_max_prob_by_area(df, area_name)
    
    # Εμφάνιση των αποτελεσμάτων
    if max_probabilities:
        st.subheader(f'Max Probabilities for {area_name}')
        for key, (value_counts, max_value, max_prob) in max_probabilities.items():
            if key == 'Month':
                st.markdown(f"**{key}**: **{months[max_value]}** with probability **{max_prob * 100:.2f}%**")
            elif key == 'Victim Descent':
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            elif key == 'Victim Sex':
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            else:
                st.markdown(f"**{key}**: **{max_value}** with probability **{max_prob * 100:.2f}%**")
            if key in ['Victim Sex', 'Crime Severity']:
                plot_probabilities(value_counts, f'{key} Distribution for {area_name}', chart_type='pie')
            else:
                plot_probabilities(value_counts, f'{key} Distribution for {area_name}')
    else:
        st.write(f"No data available for {area_name}")