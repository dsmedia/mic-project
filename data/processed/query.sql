
SELECT id,
strftime(TRY_STRPTIME(publication_date, '%b %d, %Y')::DATE,'%a., %b. %-d, %Y') as publication_date,
--coalesce(publication_date::VARCHAR, 'N/A') as publication_date,
COALESCE(location, 'N/A') as location,
COALESCE(subject, 'N/A') as subject,
COALESCE(people, 'N/A') as people,
coalesce(full_text, '[TEXT MISSING]') as full_text
from
raw.articles a
WHERE
(
(replace(section, ' ', '') IS NULL OR replace(section, ' ', '') LIKE '%Fore%')
AND NOT EXISTS (
SELECT 1
FROM staging.excludable_subjects e
JOIN (
SELECT trim(value) as subject
FROM unnest(string_split(a.subject, ';')) AS t(value)
) s ON e.subject_name = s.subject
)
AND (
a.subject IS NULL
OR EXISTS (
SELECT 1
FROM staging.relevant_subjects r
JOIN (
SELECT trim(value) as subject
FROM unnest(string_split(a.subject, ';')) AS t(value)
) s ON r.subject_name = s.subject
)
)
)
AND (
a.location IS NULL
OR EXISTS (
SELECT 1
FROM (
SELECT trim(value) as location_value
FROM unnest(string_split(a.location, ';')) AS t(value)
) loc
WHERE NOT loc.location_value IN (
'Texas', 'Long Island New York', 'Iowa', 'Louisiana', 'Ohio', 'Arizona', 'Pittsburgh Pennsylvania', 'Paso Robles California', 'Cape Cod Massachusetts', 'Liberty Island', 'Kansas', 'South Florida', 'Moscow Idaho', 'Rhode Island', 'Monterey Park California', 'Santa Clarita California', 'Tampa Bay', 'Santa Clara County California', 'East Texas', 'Virgin Islands-US', 'Henrico County Virginia', 'Bend Oregon', 'Fort Chaffee', 'Northern Mariana Islands', 'Ross Lake', 'Ventura County California', 'Macomb County Michigan', 'Arizona Trail', 'Vancouver Washington', 'Marin County California', 'Costa Mesa California', 'Los Altos California', 'Jersey Shore', 'Potomac River', 'North Charleston South Carolina', 'Bonneville Salt Flats', 'Sayreville New Jersey', 'Galveston County Texas', 'Round Rock Texas', 'Tenne ssee', 'Huntington Beach California', 'Rancho Palos Verdes California', 'Riverside County California', 'Fort William Henry', 'Willamette Valley', 'Rocky Mountains', 'Nueces County Texas', 'Visalia California', 'Great Plains', 'Milwaukee Wisconsin', 'Ma ine', 'Portland Oregon', 'California', 'Colorado', 'Los Angeles California', 'Alabama', 'Nashville Tennessee', 'Baton Rouge Louisiana', 'Hawaii', 'Washington (state)', 'San Clemente California', 'Mississippi', 'Illinois', 'Delaware', 'Des Moines Iowa', 'Detroit Michigan', 'Broward County Florida', 'Lake Mead', 'North Dakota', 'Bexar County Texas', 'Tarrant County Texas', 'Cobb County Georgia', 'Inland Empire', 'Gwinnett County Georgia', 'Finger Lakes', 'Kern County California', 'Oakland California', 'Fort Bliss', 'Redwood City California', 'San Juan River', 'Tijuana River', 'Anaheim California', 'Culver City California', 'Los Angeles Calif ornia', 'Coweta County Georgia', 'McLennan County Texas', 'Permian Basin', 'Windsor Virginia', 'Sturgis South Dakota', 'East Baton Rouge Parish Louisiana', 'Marco Island', 'North Slope', 'Priest Lake', 'Devils Lake', 'Mount Rushmore', 'Andover New Jersey', 'Morningside Park', 'Long Island Sound', 'New Jersey', 'New York City New York',
'Alabama River', 'Alameda County California', 'Alaska', 'Amarillo Texas', 'Amelia Island', 'American River', 'American Samoa', 'Ann Arbor Michigan', 'Appalachia', 'Arkansas', 'Atlanta Georgia', 'Bakersfield California', 'Baltimore Maryland', 'Bolsa Chica', 'Boston Massachusetts', 'Boulder County Colorado', 'Bristol Bay', 'Bronx New York', 'Brooklyn New York', 'Camp David', 'Cape Canaveral Florida', 'Carrizo Springs Texas', 'Cayuga Lake', 'Central Park-New York City NY', 'Central Valley', 'Charleston South Carolina', 'Charlottesville Virginia', 'Chattahoochee River', 'Chesapeake Bay', 'Chicago Illinois', 'Chicago River', 'Chula Vista California', 'City Island', 'Cleveland Ohio', 'Colleyville Texas', 'Collin County Texas', 'Colorado River', 'Colorado Springs Colorado', 'Columbia River', 'Columbus Ohio', 'Connecticut', 'Cook County Illinois', 'Crystal River', 'Death Valley', 'Detroit River', 'East Bay California', 'Eastern Kentucky', 'East Lansing Michigan', 'East Los Angeles California', 'East Palestine Ohio', 'Eastern states', 'Ellis Island', 'Escondido California', 'Everglades', 'Florida', 'Florida Keys', 'Fort Bliss', 'Fort Hancock', 'Fort McClellan', 'Fort Missoula', 'Fort Worth Texas', 'Fox Lake', 'Franklin County Ohio', 'Frio County Texas', 'Georgia', 'Governors Island', 'Gowanus Canal', 'Grand Canyon', 'Grand Rapids Michigan', 'Grand River', 'Great Lakes', 'Green River', 'Guam', 'Gulf of Mexico', 'Harlem River', 'Hart Island', 'Houston Texas', 'Hudson River', 'Hudson River Park', 'Hudson Valley', 'Iberia Parish Louisiana', 'Idaho', 'Indiana', 'Indianapolis Indiana', 'Indian River', 'Inglewood California', 'Iowa', 'Iowa City Iowa', 'Jefferson Parish Louisiana', 'Kalamazoo Michigan', 'Kansas City Missouri', 'Kenosha Wisconsin', 'Kentucky', 'Kentucky River', 'Kinney County Texas', 'Kirkland Washington', 'La Crosse Wisconsin', 'La Jolla California', 'Lafayette Louisiana', 'Lake Charles', 'Lake Erie', 'Lake George', 'Lake Huron', 'Lake Michigan', 'Lake Ontario', 'Lake Shasta', 'Lake Superior', 'Lake Tahoe', 'Laredo Texas', 'Las Colinas Texas', 'Las Vegas Nevada', 'Lexington Kentucky', 'Little Red River', 'Los Angeles County California', 'Los Angeles River', 'Louisville Kentucky', 'Lubbock Texas', 'Maine', 'Malibu California', 'Maryland', 'Massachusetts', 'Mauna Kea', 'Mauna Loa', 'Mendocino County California', 'Merced County California', 'Miami Florida', 'Miami-Dade County Florida', 'Michigan', 'Midwest states', 'Minneapolis Minnesota', 'Minnesota', 'Mississippi River', 'Mississippi Valley', 'Missouri', 'Modesto California', 'Mojave Desert', 'Montana', 'Monterey County California', 'Moreno Valley', 'Mount Rainier', 'Mount Shasta', 'Mount Tamalpais', 'Nantucket Massachusetts', 'Napa County California', 'Napa Valley', 'Nebraska', 'Nevada', 'New Hampshire', 'New Mexico', 'New Rochelle New York', 'New York', 'New York City New York', 'New York Harbor', 'Newark New Jersey', 'Newport Beach California', 'Neuse River', 'Newtown Creek', 'Niagara Falls', 'North Carolina', 'Northern California', 'Northeastern states', 'Oakland County Michigan', 'Ohio River', 'Oklahoma', 'Orlando Florida', 'Orleans Parish Louisiana', 'Pacific Grove California', 'Pacific Northwest', 'Palo Alto California', 'Parris Island', 'Pawleys Island', 'Pearl Harbor', 'Pearl River', 'Pennsylvania', 'Philadelphia Pennsylvania', 'Pine River', 'Placer County California', 'Poudre River', 'Prescott Arizona', 'Puerto Rico', 'Queens New York', 'Raleigh North Carolina', 'Randalls Island', 'Rancho Mirage California', 'Redondo Beach California', 'Rio Grande', 'Rio Grande River', 'Rio Grande Valley', 'Riverside Park', 'Robstown Texas', 'Rock Creek', 'Round Valley', 'Russian River', 'Sacramento California', 'Sacramento County California', 'Salt River', 'San Bernardino County California', 'San Bruno California', 'San Diego County California', 'San Francisco Bay', 'San Francisco California', 'San Jacinto County Texas', 'San Joaquin County California', 'San Joaquin Valley', 'San Juan Capistrano California', 'San Juan Mountains', 'San Leandro California', 'San Luis Obispo California', 'San Luis Obispo County California', 'San Mateo California', 'Sanibel Island', 'Santa Barbara California', 'Santa Clara Valley', 'Santa Cruz County California', 'Sarasota County Florida', 'Sarasota Florida', 'Savannah River', 'Seattle Washington', 'Shasta County California', 'Sheepshead Bay', 'Silicon Valley-California', 'Simi Valley California', 'Snake River', 'Solano County California', 'Sonoma County California', 'South Carolina', 'South Dakota', 'Southern California', 'Southern states', 'Southeastern states', 'Squaw Valley', 'St Louis Missouri', 'Starr County Texas', 'Stone Mountain', 'Susquehanna Valley', 'Tennessee', 'Thousand Oaks California', 'Three Mile Island', 'Trinity River', 'Tulare County California', 'United States --US', 'United States-- US', 'United States--U S', 'United States--US', 'United Sta tes--US', 'United St ates--US', 'Un ited States--US', 'Upper Klamath Lake', 'Utah', 'Uvalde County Texas', 'Vermont', 'Virginia', 'Washington DC', 'Washington Square Park', 'Webb County Texas', 'West Hollywood California', 'West Texas', 'West Virginia', 'Western states', 'White River', 'Wilmington Delaware', 'Winooski River', 'Wisconsin', 'Wyoming', 'Zapata County Texas', 'Zavala County Texas',
'Iow a', 'Boone North Carolina', 'Brazoria County Texas', 'Cook Inlet', 'Grand Island New York', 'Green Mountains', 'Iowa', 'Kentucky', 'Lake of the Woods', 'Marina del Rey California', 'Matagorda Bay', 'Memphis Tennessee', 'Midway Islands', 'New Hampshire', 'New York', 'New York City New York', 'Oregon', 'Red River', 'Saipan', 'San Francisco California', 'St Tammany Parish Louisiana', 'Straits of Florida', 'Thousand Islands', 'Tinian', 'United States--US', 'Wyoming'
)
)
)