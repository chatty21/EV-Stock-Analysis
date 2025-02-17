import requests

# Corrected URL with the right endpoint
url = "https://soccer.sportmonks.com/api/v2.0/fixtures?api_token=id6QWtJ45UxqxKqNPjHavoVfnxsywfR4u96R6kXeMPjRfKxh1s1e4Rpjlqmw&include=localTeam,visitorTeam"

response = requests.get(url)
data = response.json()

print(data)
