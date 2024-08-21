# EVCDgen - Electric Vehicle Charging Demand generator

# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.

# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.



import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt


# ## INIT ###
# (1) Run prepareData.py to generate the necessary inputs for the schedule generator
# (2) Run chargingProbability.py to generate charging probability curves
# (3) Now, you should be able to import the below files. We distinguish between the 2 'working' user groups (W) and the 'retired' (R) user group.
# (4) Run main.py --> standard run is 100 EVs.
# (5) Results are stored in the 'result' dict.



### SET ###
number_of_evs = 1000
result = {}



### IMPORT FILES ###
# Information about home to work trips
hometowork = pd.read_csv('../intermediate/hometowork.csv') # Generated by prepareData.py
workingtime = pd.read_csv('../inputs/workingtime.csv')

# Information on 'non working trips' on working days
W_non_working_trips_on_working_days = pd.read_csv('../intermediate/W_non_working_trips_on_working_days.csv')
W_trips_on_non_working_days = pd.read_csv('../intermediate/W_trips_on_non_working_days.csv')

R_non_working_trips_on_working_days = pd.read_csv('../intermediate/R_non_working_trips_on_working_days.csv')
R_trips_on_non_working_days = pd.read_csv('../intermediate/R_trips_on_non_working_days.csv')

# Charging probability curve
charging_probability = pd.read_excel('../intermediate/charging_probability.xlsx')



### HELPER FUNCTIONS ###
def selectWorkingSchedule(total_weeks):
	possible_schedules = [0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 0 , 0, 0], [1, 1, 1, 0, 0, 0, 0]
	schedule = random.choices(possible_schedules, [0.2, 0.31, 0.31, 0.06, 0.06, 0.06])[0] * total_weeks
	return schedule

# Outputs some intervals to add small variety.
def addVariety():
	number = np.random.randint(3)
	return number

# Outputs charging yes/no based on probability (e.g. 60%, 0.6)
def chargingDecision(probability):
    return random.random() < probability

# Finds the next trip
def findNextTrip(schedule, interval, stepsize=1):
    filtered = np.where(schedule == 11.0)
    filtered = filtered[0]
    grouped_trips = np.split(filtered, np.where(np.diff(filtered) != stepsize)[0]+1)
    next_trip = [0]
    
    for i in grouped_trips:
        if i[0] > interval:
            next_trip = i
            break
        
    return next_trip


### CREATE SCHEDULE ###
def createSchedule():
	total_weeks = 2
	days = 7 * total_weeks

	ev = {}
	workschedule = selectWorkingSchedule(total_weeks)
	ev_driving_velocity_per_minute = 0.70 # kilometer per minute
	ev_driving_distance_per_interval = 5 * ev_driving_velocity_per_minute
	ev_power_consumption_per_5_minutes = 0.175 * ev_driving_distance_per_interval # 17.5 kWh per 100 km --> 0.175 kWh per km * driving_distance

	ev_charging_power = 11000 # kW
	ev_charging_energy_per_5_minutes = ev_charging_power / 1000 / 12 # charging power in kW(h) divided by 12 (5-minute) intervals

	ev_battery_capacity = 50 # kWh

	ev = {
	   'weekschedule': np.zeros(0),
	   'workroutine': workschedule,
	   'leave_for_work': random.choices(hometowork['departure'].index, hometowork['departure'])[0],
	   'travel_time_to_work': random.choices(hometowork['travel'].index, hometowork['travel'])[0],
	   'time_spent_at_work': random.choices(workingtime['workingtime'].index, workingtime['workingtime'])[0],
	   'power_consumption_per_5_minutes': ev_power_consumption_per_5_minutes,
	   'battery_capacity': ev_battery_capacity,
	   'charging_power': ev_charging_power, # W
	   'soc_decline_per_5_minutes': ev_power_consumption_per_5_minutes * 100 / ev_battery_capacity,
	   'soc_increase_per_5_minutes': ev_charging_energy_per_5_minutes * 100 / ev_battery_capacity,
	   'state_of_charge': np.zeros(2016 * total_weeks),
	   'power_status': np.array(['unkown' for _ in range(2016* total_weeks)], dtype = object),
	   'power_demand_home': np.zeros(2016 * total_weeks),
	   'power_demand_work': np.zeros(2016 * total_weeks),
	   'power_demand_destination': np.zeros(2016 * total_weeks),
	   'home_charging' : random.choices(['yes', 'no'], [0.5, 0.5])[0],
	   'workplace_charging': random.choices(['yes', 'no'], [0.5, 0.5])[0],
	   'charging_boolean': np.ones(2016 * total_weeks) * True,
	   'driving_distance_per_interval' : ev_driving_distance_per_interval,
	   'distance_traveled': 0,
	   'maxSoC': 100,
	   'startSoC': 100,
	   'working': 'yes',
	   'total_weeks': total_weeks,
	   'day_of_week': [2, 3, 4, 5, 6, 7, 1] * total_weeks # Relates to day of week in original data
	   }
    
	if workschedule == [0, 0, 0, 0, 0, 0, 0]:

		ev['working'] = 'no'

	# Reset to 0
	beginning_of_next_day = np.zeros(0)

	# Loop over all days
	for day in range(days):
		# Reset to 0
		day_array = np.zeros(0)

		# Working day
		if workschedule[day] == 1:
			# If there is info from previous day, start with that:
			if beginning_of_next_day.size > 0:
				day_array = np.append(day_array, beginning_of_next_day)
                
			# Departure to work --> pad with 0's (at home)
			departure = ev['leave_for_work'] + addVariety()
			day_array = np.append(day_array, np.zeros(departure - len(beginning_of_next_day))) # Deduct whatever was left of yesterday!
    
			# Travel time to work --> driving --> append 11
			travel = ev['travel_time_to_work'] + addVariety()
			day_array = np.append(day_array, np.ones(travel)*11)
            
			# Working time --> append 10
			working = ev['time_spent_at_work'] + addVariety()
			day_array = np.append(day_array, np.ones(working)*10)
            
			# Travel back home --> driving --> append 11
			day_array = np.append(day_array, np.ones(travel)*11)
        
			# Now we are back home, are we going to do an activity by car?
			coming_home_time = len(day_array)
                
			# If we arrive home after midnight:
			if coming_home_time > 288:
				beginning_of_next_day = day_array[288:coming_home_time] # Shift to beginning of next day
				day_array = day_array[0:288] # Fix day at correct length
            
			# Make a set of potential trips, containing all non-working-trips on working days:
			potential_trips = W_non_working_trips_on_working_days
			potential_trips = potential_trips[potential_trips['RVertTijd'] > coming_home_time*5] # Select trips after coming_home_time
            
			# Define the probability that the EV is doing something (no. of potential trips / no. of actual non working trips in data set)
			probability = len(potential_trips.index) / len(W_non_working_trips_on_working_days.index)
            
			# Check if we do (1) or do not (0) an activity after work:
			ActivityAfterWork = random.choices([1,0],[probability, 1-probability])
    
			if ActivityAfterWork[0] >= 1:
				# Take a sample from the set of potential trips
				activity = potential_trips.sample().reset_index(drop=True)
				TypeOfActivity = activity['KMotiefV'][0]
				departure_time = int(round(activity['RVertTijd'][0]/5))
				travel_time = int(round(activity['RReisduur'][0]/5))
				activity_time = int(round(activity['ActDuur'][0]/5))

				at_home_time = departure_time - coming_home_time
				day_array = np.append(day_array, np.zeros(at_home_time))
				day_array = np.append(day_array, np.ones(travel_time)*11) # Driving to activity
				day_array = np.append(day_array, np.ones(activity_time)* TypeOfActivity)  # Doing the activity
				day_array = np.append(day_array, np.ones(travel_time)*11) # Driving back home
                
				coming_home_time = len(day_array)
                
				# Check if we come home after midnight
				if coming_home_time > 288:
					beginning_of_next_day = day_array[288:coming_home_time] # Shift to beginning of next day
					day_array = day_array[0:288] # Fix day at correct length
                
				# Fill array at home
				try:
					day_array = np.append(day_array,np.zeros(288-day_array.size)) # Stay home till end of day
				except:
					pass
                            
			# If NO activity after work, stay home till end of day
			else:
				try:
					day_array = np.append(day_array,np.zeros(288-day_array.size))
				except:
					pass

		# NON-Working day
		if workschedule[day] == 0:
			# Reset to zero
			day_array = np.zeros(288)

			# Check if individual is working
			if ev['working'] == 'yes':
				trips = W_trips_on_non_working_days.query('Weekdag == '+str(ev['day_of_week'][day])+'')

			# No, retired
			else:
				trips = R_trips_on_non_working_days.query('Weekdag == '+str(ev['day_of_week'][day])+'')

			# Select unique person
			opids = trips['OPID'].unique()
			opid = random.choice(opids)
            
			# Select unique person's day
			day_data = trips[trips['OPID'] == opid]
			day_data = day_data.reset_index()
            
			# Separate the different trips on the day --> 'verplaatsingen'
			verplaatsingen = [v for k, v in day_data.groupby('VerplID')]
            
			# Go through all 'verplaatsingen' (trips) inside the 'personday'
			for i in range(len(verplaatsingen)):
				verplaatsingen[i] = verplaatsingen[i].reset_index()

				# We gaan vertrekken
				uur_vertrek = verplaatsingen[i]['VertUur'][0]
				minuut_vertrek = verplaatsingen[i]['VertMin'][0]
				interval_vertrek = int(round( (uur_vertrek*60 + round(minuut_vertrek)  ) / 5 ))

				# Hoe lang reizen?
				uur_aankomst = verplaatsingen[i]['AankUur'][0]
				minuut_aankomst = verplaatsingen[i]['AankMin'][0]
				interval_aankomst = int(round(( uur_aankomst*60 + round(minuut_aankomst) ) / 5))
                
				reistijd = interval_aankomst - interval_vertrek

				# Vul het schema tussen interval_vertrek en interval_aankomst met het vervoermiddel:
				day_array[interval_vertrek:interval_aankomst] = 11 #
                        
				# Wat gaan we doen?
				activiteit = verplaatsingen[i]['KMotiefV'][0]

				# Tot hoelaat?
				duur = int(round(verplaatsingen[i]['ActDuur'][0] / 5)) # minuten -> 5-minute interval
                
				# Normal activity (actually doing something at a location)
				if verplaatsingen[i]['Toer'][0] == 0:
					day_array[interval_aankomst:interval_aankomst+duur] = activiteit  # Doing activity
					day_array[interval_aankomst+duur:interval_aankomst+duur+reistijd] = 11  # Driving
                
				# Toer: special! (driving around without stopping at a location --> start and and at home.)
				if verplaatsingen[i]['Toer'][0] == 1:
					duur = int(round(verplaatsingen[i]['Reisduur'][0]))
					if duur > 12: # If tour takes very long (corner case)
						duur = 10
                        
					day_array[interval_vertrek:interval_vertrek+duur] = 11 # driving
                        
		# Append day to the schedule
		ev['weekschedule'] = np.append(ev['weekschedule'], day_array)

	return ev



### CHARGING MODEL ###
def doCharging(ev):
	total_weeks = ev['total_weeks']
	interval = 0
	ev['state_of_charge'][0] = ev['startSoC'] # Start with SoC of 100
	ChargingBoolean = False

	# Loop through schedule (activity can be at home, driving, doing activity etc.)
	for activity in ev['weekschedule']:

		if interval == 0:  # Start up, start parked at home
			ev['state_of_charge'][interval] = ev['state_of_charge'][0]
			ev['power_status'][interval] = 'parkedAtHomeNotCharging'

		if interval != 0:  # We are somewhere in the week

			# Debug corner cases
			if ev['state_of_charge'][interval] < 0.0:
				print("DEBUG: interval " + str(interval))
				break


        
			##### AT HOME #####
			if activity == 0.0: # We are at home
				ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1]
				ev['power_status'][interval] = 'parkedAtHomeNotCharging'

				# Bug fix to save last corner cases. If SoC for whatever reason is below 0, put it to 20.
				if ev['state_of_charge'][interval] <= 0:
					ev['state_of_charge'][interval] = 20
	
				# Can we charge at home?
				if ev['home_charging'] == 'yes':
					# Check if we have 'change of mode' --> ensures we don't reset the charging choice every interval. Once we decide we charge, we keep charging and skip this stuff.
					if ev['weekschedule'][interval-1] != activity:
						print('INTERVAL ' + str(interval) + ' change of mode - home')
	
						# Decide if we want to charge based on charging-soc curve
						local_soc = int(round(ev['state_of_charge'][interval]))
						local_probability = charging_probability[charging_probability['SoC']==local_soc]['home'][local_soc] / 100 # Percentage so divide with 100
						ChargingBoolean = chargingDecision(local_probability)
	
						print('INTERVAL ' + str(interval) + " SoC = " + str(local_soc) + " Charging Prob = " + str(local_probability) + " Decision = " + str(ChargingBoolean) + "\n")
	
						ev['charging_boolean'] = np.ones(2016 * total_weeks) * ChargingBoolean
	    
						# Check if we can reach our next destination, if not charge anyway (even if previously we did not!)
						nexttrip = findNextTrip(ev['weekschedule'], interval)  # (add 2 for safety margin)
						if (len(nexttrip)+2) * ev['soc_decline_per_5_minutes'] > ev['state_of_charge'][interval]:
							print('cannot reach next destination so we have to charge')
							ev['charging_boolean'] = np.ones(2016* total_weeks) * True
							ChargingBoolean = True
	
					# Charging process --> check if we don't exceed battery capacity (SoC > 100%) and that we should charge (boolean)
					if ev['state_of_charge'][interval] + ev['soc_increase_per_5_minutes'] < ev['state_of_charge'][0] and ChargingBoolean == True:
                        
						# Update SoC, status and power demand curve
						ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1] + ev['soc_increase_per_5_minutes']
						ev['power_status'][interval] = 'chargingAtHomePrivate'
						ev['power_demand_home'][interval] = ev['charging_power']

					# No charging now;
					if ChargingBoolean == False:
						pass
	
	
				# We don't have a home charger. If SoC < 20% --> charge public CP. Check if we can make the next trip, if not, use public CP;
				if ev['home_charging'] == 'no':
					# ChargingBoolean = False
	
					# Check if we have 'change of mode' --> ensures we don't reset the charging choice every interval. Once we decide we charge, we keep charging and skip this stuff.
					if ev['weekschedule'][interval-1] != activity:
						print('INTERVAL ' + str(interval) + ' change of mode - public at home')
	
						# Decide if we want to charge based on charging-soc curve
						local_soc = int(round(ev['state_of_charge'][interval]))
						local_probability = charging_probability[charging_probability['SoC']==local_soc]['public'][local_soc] / 100 # Percentage so divide with 100
						ChargingBoolean = chargingDecision(local_probability)
	
						print('INTERVAL ' + str(interval) + " SoC = " + str(local_soc) + " Charging Prob = " + str(local_probability) + " Decision = " + str(ChargingBoolean) + "\n")
	                        
						ev['charging_boolean'] = np.ones(2016* total_weeks) * ChargingBoolean
	                    
						# Check if we can reach our next destination, if not charge anyway (even if previously we did not!)
						nexttrip = findNextTrip(ev['weekschedule'], interval)
						if (len(nexttrip)+2) * ev['soc_decline_per_5_minutes'] > ev['state_of_charge'][interval]:
							print('INTERVAL ' + str(interval) + ' Cannot reach next destination so we have to charge')
							ev['charging_boolean'] = np.ones(2016* total_weeks) * True
							ChargingBoolean = True

						if ev['state_of_charge'][interval] <= 20:
							ChargingBoolean = True
	                    
							# Somehow the charging boolean is set to True here when start at home
	
					# Charging process --> check if we don't exceed battery capacity (SoC > 100%) and that we should charge (boolean)
					if ev['state_of_charge'][interval] + ev['soc_increase_per_5_minutes'] < ev['maxSoC'] and ChargingBoolean == True:
						# print('INTERVAL ' + str(interval) + ' Charge at PUBLIC when at home since no home charging but we need energy!')
						ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1] + ev['soc_increase_per_5_minutes']
						ev['power_status'][interval] = 'chargingNearHomePublic'
						ev['power_demand_destination'][interval] = ev['charging_power']



			##### DRIVING #####
			if activity == 11.0: # Driving, just discharge
				ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1] - ev['soc_decline_per_5_minutes']
				ev['power_status'][interval] = 'discharging'


			##### AT WORK #####
			if activity == 10.0: # At work
				ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1]
				ev['power_status'][interval] = 'parkedAtWorkNotCharging'
                
				# Can we charge at work?
				if ev['workplace_charging'] == 'yes':
                    
					# Check if we have 'change of mode' --> ensures we don't reset the charging choice every interval. Once we decide we charge, we keep charging and skip this stuff.
					if ev['weekschedule'][interval-1] != activity:
						print('INTERVAL ' + str(interval) + ' Change of mode - work')

						# Decide if we want to charge based on charging-soc curve
						local_soc = round(ev['state_of_charge'][interval])
						local_probability = charging_probability[charging_probability['SoC']==local_soc]['work'][local_soc] / 100 # percentage so divide with 100
                        
						ChargingBoolean = chargingDecision(local_probability)
                        
						print("INTERVAL " + str(interval) + " SoC = " + str(local_soc) + " Charging Prob = " + str(local_probability) + " Decision = " + str(ChargingBoolean) + "\n")
                        
						ev['charging_boolean'] = np.ones(2016* total_weeks) * ChargingBoolean
                    
                    
					# Check if we can reach our next destination, if not charge anyway (even if previously we did not!)
					nexttrip = findNextTrip(ev['weekschedule'], interval)
                    
					if (len(nexttrip)+2)* ev['soc_decline_per_5_minutes'] > ev['state_of_charge'][interval]:
						print('cannot reach next destination so we have to charge')
						ev['charging_boolean'] = np.ones(2016* total_weeks) * True
						ChargingBoolean = True

                    
					# Charging process
					if ev['state_of_charge'][interval] + ev['soc_increase_per_5_minutes'] < ev['maxSoC'] and ChargingBoolean == True:
						ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1] + ev['soc_increase_per_5_minutes']
						ev['power_status'][interval] = 'chargingAtWorkplace'
						ev['power_demand_work'][interval] = ev['charging_power']
                
                

				if ev['workplace_charging'] == 'no':
					ChargingBoolean=False

					# Check if we have 'change of mode' --> ensures we don't reset the charging choice every interval. Once we decide we charge, we keep charging and skip this stuff.
					if ev['weekschedule'][interval-1] != activity:
						print('INTERVAL ' + str(interval) + ' Change of mode - work')
                        
						# Check if we can reach our next destination, if not charge anyway (even if previously we did not!)
						nexttrip = findNextTrip(ev['weekschedule'], interval)
	                        
						if (len(nexttrip)+2)* ev['soc_decline_per_5_minutes'] > ev['state_of_charge'][interval]:
							print('cannot reach next destination so we have to charge')
							ev['charging_boolean'] = np.ones(2016* total_weeks) * True
							ChargingBoolean = True

						if ev['state_of_charge'][interval] <= 20:
							ChargingBoolean = True

					# Charging process
					if ev['state_of_charge'][interval] + ev['soc_increase_per_5_minutes'] < ev['maxSoC'] and ChargingBoolean == True:
						ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1] + ev['soc_increase_per_5_minutes']
						ev['power_status'][interval] = 'chargingNearWorkPublic'
						ev['power_demand_destination'][interval] = ev['charging_power']


			##### DOING AN ACTIVITY #####
			if activity >= 1.0 and activity <= 9.0: # OP is somewhere else --> KMotiefV is 1-9
				ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1]
				ev['power_status'][interval] = 'parkedAtActivityNotCharging'


				# Check if we can reach our next destination, if not charge at pulic CP;
				nexttrip = findNextTrip(ev['weekschedule'], interval)
				if (len(nexttrip)+2)* ev['soc_decline_per_5_minutes'] > ev['state_of_charge'][interval]:
					print('cannot reach next destination so we have to charge')
					ev['charging_boolean'] = np.ones(2016* total_weeks) * True
					ChargingBoolean = True
                    
					# Charging process
					if ev['state_of_charge'][interval] + ev['soc_increase_per_5_minutes'] < ev['maxSoC'] and ChargingBoolean == True:
						print('INTERVAL ' + str(interval) + ' Charge at DESTINATION!')
						ev['state_of_charge'][interval] = ev['state_of_charge'][interval-1] + ev['soc_increase_per_5_minutes']
						ev['power_status'][interval] = 'chargingPublicAtDestination'
						ev['power_demand_destination'][interval] = ev['charging_power']

		# Update interval
		interval += 1

	# Calculate traveled distance with EV in this week
	ev['distance_traveled'] = np.count_nonzero(ev['weekschedule'] == 11.0) * ev['driving_distance_per_interval'] # no. of intervals driving * distance per interval

	return ev


### PLOT SINGLE EV ###
def plotSingleEV(ev):
	fig, axs = plt.subplots(2,1,figsize=(6, 3))
	axs[0].plot(ev['state_of_charge'], drawstyle='steps', color='black')
	axs[0].set_ylim(0, 100)
	axs[0].set_xlim(0, 2016*2)
	axs[0].set_ylabel('SoC [%]')
	axs[0].grid()
	    
	axs[1].plot(ev['power_demand_home']/1000, drawstyle='steps', label='Home')
	axs[1].plot(ev['power_demand_work']/1000, drawstyle='steps', label='Work')
	axs[1].plot(ev['power_demand_destination']/1000, drawstyle='steps', label='Public')
	axs[1].set_ylim(0, 12)
	axs[1].set_xlim(0, 2016*2)
	axs[1].set_ylabel('Power [kW]')
	axs[1].grid()
	axs[1].legend()
	return




### RUN SIMULATION
for evs in range(number_of_evs):
	print('Generating EV schedule number '+str(evs))
	result[evs] = createSchedule()

	print('Running charging module for EV number ' + str(evs))
	result[evs] = doCharging(result[evs])


### PLOT A SINGLE EV
single_ev = result[0]
plotSingleEV(single_ev)


### PLOT ACCUMULATED POWER
total_power_work = np.zeros(len(result[0]['weekschedule']))
total_power_residential = np.zeros(len(result[0]['weekschedule']))
total_power_public = np.zeros(len(result[0]['weekschedule']))

for ev in result:
    total_power_work = total_power_work + result[ev]['power_demand_work']
    total_power_residential = total_power_residential + result[ev]['power_demand_home']
    total_power_public = total_power_public + result[ev]['power_demand_destination']


fig, ax = plt.subplots(figsize=(6, 2))
ax.plot(total_power_residential/1000, label='Home')
ax.plot(total_power_work/1000, label = 'Work')
ax.plot(total_power_public/1000, label ="Public")
ax.set_xlim(0,len(result[0]['weekschedule']))
ax.set_ylim(0)
ax.set_xlabel("Time interval [-]")
ax.set_ylabel("Power [kW]")
plt.legend()
plt.grid()