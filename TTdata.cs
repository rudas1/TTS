namespace NeuralNetwork
{
    class TTdata
    {
        private string Class;   //klase pvz 3A
        private string subject; //dalykas pvz Matematika
        private string teacher; //mokytojas Petras Petraitis
        private int count;  //pamoku per savaite pvz 6

        private int subjectID;
        private int classID;

        public TTdata()     //Constructor
        {
            
        }

        //Set value

        public void SetClass(string value)
        {
            if(value.Length == 2) Class = value;
        }
        public void SetSubject(string value)
        {
            subject = value;
        }
        public void SetTeacher(string value)
        {
            teacher = value;
        }
        public void SetCount(int value)
        {
            count = value;
        }
        public void SetSubjectID(int value)
        {
            subjectID = value;
        }
        public void SetClassID(int value)
        {
            classID = value;
        }

        //Get value

        public string GetClass()
        {
            return Class;
        }
        public string GetSubject()
        {
            return subject;
        }
        public string GetTeacher()
        {
            return teacher;
        }
        public int GetCount()
        {
            return count;
        }
        public int GetSubjectID()
        {
            return subjectID;
        }
        public int GetClassID()
        {
            return classID;
        }
    }
}
