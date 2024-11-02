'''import csv

# Define the list to store users from Dublin
users_in_dublin = []

# Read the CSV file with UTF-8 encoding
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        location = row['location'].strip().lower()
        # Check if the user is from Dublin
        if 'dublin' in location:
            users_in_dublin.append({
                'login': row['login'],
                'followers': int(row['followers'])
            })

# Sort users based on followers in descending order
top_users = sorted(users_in_dublin, key=lambda x: x['followers'], reverse=True)

# Extract the top 5 user logins
top_5_logins = [user['login'] for user in top_users[:5]]

# Print the result as a comma-separated list
print("Top 5 users in Dublin with the highest number of followers:")
print(', '.join(top_5_logins))'''
'''import csv

# Define the list to store users from Dublin
users_in_dublin = []

# Read the CSV file with UTF-8 encoding
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        location = row['location'].strip().lower()
        # Check if the user is from Dublin
        if 'dublin' in location:
            users_in_dublin.append({
                'login': row['login'],
                'created_at': row['created_at']
            })

# Sort users based on created_at in ascending order
earliest_users = sorted(users_in_dublin, key=lambda x: x['created_at'])

# Extract the first 5 user logins
earliest_5_logins = [user['login'] for user in earliest_users[:5]]

# Print the result as a comma-separated list
print("5 earliest registered GitHub users in Dublin:")
print(','.join(earliest_5_logins))'''
'''import csv
from collections import Counter

# Define a list to store licenses
licenses_count = Counter()

# Read the CSV file with UTF-8 encoding
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        license_name = row['license_name'].strip()
        if license_name:  # Ignore missing licenses
            licenses_count[license_name] += 1

# Get the 3 most common licenses
most_common_licenses = licenses_count.most_common(3)

# Extract the license names
top_3_licenses = [license_name for license_name, _ in most_common_licenses]

# Print the result as a comma-separated list
print("3 most popular licenses among users in Dublin:")
print(','.join(top_3_licenses))'''
'''import csv
from collections import Counter

# Load user data from users.csv
companies = []

with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        company = row['company'].strip().upper()  # Clean up company names
        if company:  # Only add non-empty values
            companies.append(company)

# Count the occurrences of each company
company_counts = Counter(companies)

# Get the company with the highest count
most_common_company = company_counts.most_common(1)[0][0]

# Print the result
print(f"The majority of these developers work at: {most_common_company}")

'''
'''import csv
from collections import Counter

# Initialize a Counter to keep track of languages
language_counter = Counter()

# Read the repositories.csv file
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        language = row['language'].strip()
        if language:  # Exclude empty or missing languages
            language_counter[language] += 1

# Get the most common language
most_common_language = language_counter.most_common(1)

# Print the result
if most_common_language:
    print(most_common_language[0][0])  # This will print the most popular language
else:
    print("No languages found in the data.")

'''
'''import csv
from collections import defaultdict

def calculate_most_popular_language(file_path: str) -> str:
    # Define a dictionary to store total stars and repository count per language
    language_stats = defaultdict(lambda: {'stars': 0, 'repos': 0})

    # Read the CSV file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # Get the language and stargazers_count field
            language = row.get('language', '').strip()
            stars = row.get('stargazers_count', '0').strip()

            # Only process if language and stars are available
            if language and stars.isdigit():
                language_stats[language]['stars'] += int(stars)
                language_stats[language]['repos'] += 1

    # Calculate average stars for each language
    average_stars_per_language = {
        language: stats['stars'] / stats['repos']
        for language, stats in language_stats.items()
        if stats['repos'] > 0
    }

    # Find the language with the highest average stars
    if average_stars_per_language:
        most_popular_language = max(average_stars_per_language, key=average_stars_per_language.get)
        return most_popular_language
    else:
        return "No language data found."

if __name__ == "__main__":
    # Specify the path to your repositories.csv file
    repos_file_path = 'repositories.csv'
    popular_language = calculate_most_popular_language(repos_file_path)
    print("The most popular language based on average stars:", popular_language)

'''
'''import csv
from collections import Counter
from datetime import datetime

# Define the list to store programming languages
languages = []

# Read the CSV file with UTF-8 encoding
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    # Iterate through the rows in the CSV
    for row in reader:
        # Parse the created_at field
        created_at = row.get('created_at', '').strip()
        
        # Convert the date string to a datetime object
        if created_at:
            user_join_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            
            # Check if the user joined after 2020
            if user_join_date.year > 2020:
                # Get the language field and clean it up
                language = row.get('language', '').strip()
                if language:
                    languages.append(language)

# Count the occurrence of each language
language_counts = Counter(languages)

# Find the two most common languages
most_common_languages = language_counts.most_common(2)

# Print the second most common language
if len(most_common_languages) >= 2:
    print(most_common_languages[1][0])  # Second most common language
else:
    print("Not enough language data found.")'''
'''import csv

# Define a list to store users and their leader strength
leader_strengths = []

# Read the CSV file with UTF-8 encoding
with open('users.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        # Get followers and following counts
        followers = int(row.get('followers', 0).strip())
        following = int(row.get('following', 0).strip())
        
        # Calculate leader strength
        leader_strength = followers / (1 + following)
        
        # Store the user's login and their leader strength
        leader_strengths.append((row.get('login', ''), leader_strength))

# Sort users by leader strength in descending order
leader_strengths.sort(key=lambda x: x[1], reverse=True)

# Get the top 5 users
top_5_leaders = [login for login, strength in leader_strengths[:5]]

# Print the result as a comma-separated list
print(','.join(top_5_leaders))
'''
'''import pandas as pd

# Read the CSV file with UTF-8 encoding
df = pd.read_csv('users.csv')

# Filter users from Dublin
dublin_users = df[df['location'].str.contains('dublin', case=False, na=False)]

# Extract the followers and public_repos columns
followers = dublin_users['followers']
public_repos = dublin_users['public_repos']

# Calculate the correlation
correlation = followers.corr(public_repos)

# Print the correlation rounded to 3 decimal places
print(f"{correlation:.3f}")
''''''
import pandas as pd
import statsmodels.api as sm

# Read the CSV file with UTF-8 encoding
df = pd.read_csv('users.csv')

# Filter users from Dublin
dublin_users = df[df['location'].str.contains('dublin', case=False, na=False)]

# Prepare the data for regression
X = dublin_users['public_repos']  # Independent variable
y = dublin_users['followers']      # Dependent variable

# Add a constant to the independent variable
X = sm.add_constant(X)

# Run the regression
model = sm.OLS(y, X).fit()

# Get the slope coefficient for public_repos
slope = model.params['public_repos']

# Print the slope rounded to 3 decimal places
print(f"{slope:.3f}")
'''
'''import pandas as pd

# Read the CSV file with UTF-8 encoding
df = pd.read_csv('repositories.csv')

# Create binary columns for 'projects_enabled' and 'wiki_enabled'
df['projects_enabled'] = df['has_projects'].apply(lambda x: 1 if x else 0)
df['wiki_enabled'] = df['has_wiki'].apply(lambda x: 1 if x else 0)

# Calculate the correlation between projects and wiki
correlation = df['projects_enabled'].corr(df['wiki_enabled'])

# Print the result rounded to 3 decimal places
print(f"{correlation:.3f}")'''
'''import pandas as pd

# Read the CSV file with UTF-8 encoding
df = pd.read_csv('users.csv')

# Calculate the average following for hireable users
hireable_avg = df[df['hireable'] == True]['following'].mean()

# Calculate the average following for non-hireable users
non_hireable_avg = df[df['hireable'] == False]['following'].mean()

# Compute the difference
average_difference = hireable_avg - non_hireable_avg

# Print the result rounded to 3 decimal places
print(f"{average_difference:.3f}")'''
'''import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def analyze_bio_followers_correlation(users_csv_path='users.csv'):
    # Read the data
    df = pd.read_csv(users_csv_path)
    
    # Filter out rows without bios
    df = df[df['bio'].notna() & (df['bio'] != '')]
    
    # Calculate bio length in Unicode characters
    df['bio_length'] = df['bio'].str.len()
    
    # Prepare data for regression
    X = df['bio_length'].values.reshape(-1, 1)
    y = df['followers'].values
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Get the slope rounded to 3 decimal places
    slope = round(model.coef_[0], 3)
    
    # Print debug information
    print(f"Number of users with bios: {len(df)}")
    print(f"Bio length range: {df['bio_length'].min()} to {df['bio_length'].max()}")
    print(f"Followers range: {df['followers'].min()} to {df['followers'].max()}")
    print(f"R-squared: {model.score(X, y):.3f}")
    
    return slope

# Calculate the regression slope
result = analyze_bio_followers_correlation()
print(f"\nRegression slope: {result:.3f}")'''
'''import csv
from collections import Counter
from datetime import datetime

# Counter to store the number of repositories created by each user on weekends
weekend_repo_counts = Counter()

# Open the repositories.csv file and read data
with open('repositories.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        created_at = row.get('created_at', '')
        if created_at:
            # Convert created_at string to a datetime object
            created_date = datetime.fromisoformat(created_at[:-1])  # Remove 'Z' and convert
            
            # Check if the day is Saturday (5) or Sunday (6)
            if created_date.weekday() in [5, 6]:
                user_login = row['login']
                weekend_repo_counts[user_login] += 1  # Increment the count for the user

# Get the top 5 users who created the most repositories on weekends
top_users = weekend_repo_counts.most_common(5)

# Extract the logins of the top users
top_logins = [user[0] for user in top_users]

# Output the top users' logins as a comma-separated string
print(','.join(top_logins))
'''
'''
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the CSV file
csv_file = 'users.csv'  # Update with your file path if necessary

# Load the data
df = pd.read_csv(csv_file)

# Filter to only users who have a bio
df = df[df['bio'].notna() & (df['bio'] != '')]

# Count the number of words in each bio (split by whitespace)
df['bio_word_count'] = df['bio'].apply(lambda x: len(x.split()))

# Prepare data for linear regression
X = df['bio_word_count'].values.reshape(-1, 1)  # Predictor: bio word count
y = df['followers'].values  # Response: number of followers

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Calculate the slope (coefficient)
slope = round(model.coef_[0], 3)

# Display the slope
print(f"Regression slope of followers on bio word count: {slope}")

''''''
import pandas as pd

# Load the CSV file with repository data
df = pd.read_csv('repositories.csv')  # Replace with the correct path

# Ensure the 'has_projects' and 'has_wiki' columns are boolean
df['has_projects'] = df['has_projects'].astype(bool)
df['has_wiki'] = df['has_wiki'].astype(bool)

# Calculate the correlation between 'has_projects' and 'has_wiki'
correlation = df['has_projects'].corr(df['has_wiki'])

# Output the correlation rounded to three decimal places
print(f"Correlation between projects and wiki enabled: {correlation:.3f}")
'''
