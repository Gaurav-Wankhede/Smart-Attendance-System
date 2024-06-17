Step 1: Check your Python Environment properly

Step 2: Install all packages using this code
	install -r requirements.txt

Step 3: use this command to run Streamlit
	Make Sure You are in the project's path

	streamlit run app.py 
	
	replace app.py if you changed file name

Step 4: In Navigation in Left side

	Click on Extract Zip to extract your pre tested dataset images

Step 5: Click on Add Student 
	if you want to add new student
	enter Name in Student's name
	enter Roll No in Student's roll number
	click on Capture Frame
	then Camera will Open 
	and will start scanning person's face 
	once complete it will close automatically

	Important Keywords:
		'q' : to exit from Frame

Step 6:Click on Train Model
	this is most important because without this we cant make proper prediction
	Click on Train the Model
	It will take time 
	during Epoch, here it will engineer 50 Epochs

Step 7: Click on Mark Attendance
	To check proper Prediction 
	And marking Attendance we will use it.
	Click on Mark your attendance
	New window will open
	if not open directly
	just check in your taskbar
	if still not open
	go to terminal and use Ctrl+C
	and rerun streamlit run app.py
	then again click on Mark your attendance
	
	Important keywords:

		'a' : To mark captured attendance
		'q' : To exit from frame
Step 8: To check all Attendance
	click on Check Attendance
	here it will show all attendance
	by checking name and date columns 
	Here it also perform remove duplicate, it is important because on that particular date, Student can mark only 1 attendance.

And Boom its Done