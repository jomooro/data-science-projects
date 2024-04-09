import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def perform_ab_test(data, alternative='two-sided', alpha=0.05):
    # Split data into control and treatment groups
    control_group = data[data['group'] == 'control']['converted']
    treatment_group = data[data['group'] == 'treatment']['converted']
    
    # Perform independent t-test
    t_statistic, p_value = stats.ttest_ind(control_group, treatment_group, alternative=alternative)
    
    # Calculate confidence intervals for both groups
    ci_control = stats.norm.interval(1 - alpha, loc=np.mean(control_group), scale=stats.sem(control_group))
    ci_treatment = stats.norm.interval(1 - alpha, loc=np.mean(treatment_group), scale=stats.sem(treatment_group))
    
    # Calculate statistical power
    effect_size = np.abs(np.mean(control_group) - np.mean(treatment_group)) / np.sqrt((np.std(control_group) ** 2 + np.std(treatment_group) ** 2) / 2)
    power = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) - effect_size)
    
    return t_statistic, p_value, ci_control, ci_treatment, power

def visualize_data(data):
    sns.countplot(x='group', hue='converted', data=data)
    plt.title('Conversion Counts by Group')
    plt.xlabel('Group')
    plt.ylabel('Count')
    plt.legend(title='Converted', labels=['No', 'Yes'])
    plt.show()

def main():
    # Load data from CSV file
    data = pd.read_csv('ab_data.csv')
    
    # Visualize the data
    visualize_data(data)
    
    print("\n=== A/B Testing Analysis Results ===")
    
    # Perform A/B test
    t_statistic, p_value, ci_control, ci_treatment, power = perform_ab_test(data)
    
    # Print results
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)
    print("Confidence Interval (Control Group):", ci_control)
    print("Confidence Interval (Treatment Group):", ci_treatment)
    print("Statistical Power:", power)
    
    # Check significance level
    alpha = 0.05
    if p_value < alpha:
        print("Result is statistically significant. There is a difference between the groups.")
    else:
        print("Result is not statistically significant. There is no difference between the groups.")

if __name__ == "__main__":
    main()
