import csv


def create_csv(file_name):


	fields = ['prn', 'epoch_time', 'sv_clock_bias', 'sv_clock_drift','sv_clock_drift_rate','iode','correction_radius_sine','mean_motion', 'mean_anomaly',
			'correction_latitude_cosine','essentricity','correction_latitude_sine','sqrt_semi_major_axis','time_of_ephemeris','correction_inclination_cosine','OMEGA',
			'correction_inclination_sine','inclination','correction_radius_cosine','omega','OMEGA_dot','inclination_rate','codes','gps_week','l2_p_data_flag',
			'sv_accuracy','sv_health','tgd','iodc','t_tx', 'fit_interval'] 



	filename = file_name
	lines = tuple(open("ephemeris_data/"+filename, 'r'))
	params=[]
	all_data_appended = []
	rows=[]


	last_frame_index = len(lines)-8
	offset = 0


	with open("csv_files/"+filename+".csv", 'w') as csvfile: 
		# creating a csv writer object 
		csvwriter = csv.writer(csvfile) 
		
		# writing the fields 
		csvwriter.writerow(fields) 


		for offset in range(0,last_frame_index,8):

			for i in range( offset + 7, offset + 15 ):
				# print(i,"--",lines[i])
				params = params+lines[i].replace('D','E').strip('\n').split(' ')

			params = list(filter(None, params)) # fastest

			# print("--fields length",len(fields))
			params[1] = params[1] + " " + params[2] + " " + params[3] + " " + params[4] + " " + params[5] + " " + params[6]
			del params[2:7]
			# print("--params--",params)
			all_data_appended = all_data_appended + params

			epoch_time = params[1]
			epoch_time == epoch_time.split(' ')
			epoch_time = str(epoch_time[0])+str(epoch_time[1])+'-'+str(epoch_time[3])+str(epoch_time[4])+'-'+str(epoch_time[6])+str(epoch_time[7])+' '+str(epoch_time[9])+str(epoch_time[10])+':'+str(epoch_time[12])+str(epoch_time[13])+':'+str(epoch_time[15])+str(epoch_time[16])
			params[1] = epoch_time

			print("THIS IS TIME__== ", epoch_time)

			csvwriter.writerow(params)

			prn = params[0]
			

			sv_clock_bias = params[2]
			sv_clock_drift = params[3]
			sv_clock_drift_rate = params[4]
			iode =  params[5]
			correction_radius_sine =  params[6]
			mean_motion = params[7]		# differential value of n
			mean_anomaly = params[8]
			correction_latitude_cosine =  params[9]
			essentricity =  params[10]
			correction_latitude_sine =  params[11]
			sqrt_semi_major_axis =  params[12]
			time_of_ephemeris =  params[13]
			correction_inclination_cosine =  params[14] 
			OMEGA =  params[15]
			correction_inclination_sine =  params[16] 
			inclination = params[17]
			correction_radius_cosine = params[18]
			omega = params[19]
			OMEGA_dot = params[20]
			inclination_rate = params[21]
			codes = params[22]
			gps_week = params[23]
			l2_p_data_flag = params[24]
			sv_accuracy = params[25]
			sv_health = params[26]
			tgd = params[27]
			iodc = params[28]
			t_tx = params[29]
			fit_interval = params[30]
			params = []

		# rows = rows + zip(prn,epoch_time, sv_clock_bias, sv_clock_drift, sv_clock_drift_rate, iode, correction_radius_sine, mean_motion, mean_anomaly, correction_latitude_cosine,
		# 					essentricity, correction_latitude_sine, sqrt_semi_major_axis, time_of_ephemeris, correction_inclination_cosine, OMEGA, correction_inclination_sine, 
		# 					inclination, correction_radius_cosine, omega, OMEGA_dot, inclination_rate, codes, gps_week, l2_p_data_flag, sv_accuracy, sv_health, tgd, iodc, t_tx, 
		# 					fit_interval )

		# print(float(sv_clock_bias))


# create_csv()
