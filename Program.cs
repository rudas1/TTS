using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    static class Program
    {
        static void SortNet(ref List<NN> list, int populationSize)
        {
            NN temporary;

            for (int i = 1; i < populationSize; i++)
            {
                if (list[i].CompareTo(list[i - 1]) == -1)
                {
                    temporary = list[i - 1];
                    list[i - 1] = list[i];
                    list[i] = temporary;
                }
            }
        }

        static void Update(ref int generation, ref List<NN> nets, ref int populationSize, int[] layers)
        {
            SortNet(ref nets, populationSize);  //surusiuoja didejancia tvarka 

            for (int i = 0; i < populationSize / 2; i++)    //puse zemiausia arba auksciausi NN
            {
                nets[i] = new NN(nets[i + (populationSize / 2)]); //sukuria kopija tokia paties bet analogiskai geresnio NN
                nets[i].Mutate();

                nets[i + (populationSize / 2)] = new NN(nets[i + (populationSize / 2)]);    //geriausi NN is naujo perdaromi identiskais
            }

            for (int i = 0; i < populationSize; i++)    //kiekvienas NN
            {
                nets[i].SetFitness(0f); //nustato pagrinda kokybes lygiui
            }

            generation++; //generacija padideja
        }

        public static float ClosestTo(this IEnumerable<float> collection, float target)
        {
            // NB Method will return int.MaxValue for a sequence containing no elements.
            // Apply any defensive coding here as necessary.
            float closest = float.MaxValue;
            float minDifference = float.MaxValue;
            foreach (float element in collection)
            {
                float difference = Math.Abs(element - target);
                if (minDifference > difference)
                {
                    minDifference = difference;
                    closest = element;
                }
            }

            return closest;
        }

        static void InitNN(ref List<NN> nets, ref int populationSize, int[] layers)   //NN apsirasymas 
        {
            if (populationSize % 2 != 0) //jeigu nelyginis, paverciamas lyginiu
            {
                populationSize--;
            }

            for (int i = 0; i < populationSize; i++)    //kiekvienam NN
            {
                NN net = new NN(layers);    //naujas net su layers
                net.Mutate();   //mutuojamas
                nets.Add(net);  //pridedamas prie bendro NN saraso
            }
        }

        static void FeedFirst(ref List<NN> nets, int populationSize, float[] inputs) //maitinimas 
        {
            List<float> perfectOutputs = new List<float>();

            for (int i = 0; i < 5; i++)
            {
                if (inputs[i * 3 + 2] != 0f)
                {
                    perfectOutputs.Add(i);
                }
            }

            for (int i = 0; i < populationSize; i++)    //kiekvienas NN
            {
                float output = (nets[i].FeedForward(inputs)[0] + 1f) * 2.5f;   //maitinami vieni duomenys

                float nearest = ClosestTo(perfectOutputs, output);

                //nets[i].AddFitness(1f - (Math.Abs(output) - nearest));    //pridedama kokybes uz tiksluma

                if (inputs[(int)nearest * 3 + 2] != 8f)
                {
                    nets[i].AddFitness(inputs[(int)nearest * 3] * -0.025f);  //dar pridedama
                }
                else
                {
                    nets[i].AddFitness(-0.05f);
                }

                nets[i].AddFitness(inputs[(int)nearest * 3 + 2] * 0.05f);               //ir dar truputi pridedama //TODO koreguoti

                if (inputs[(int)nearest * 3 + 1] > 1)
                {
                    nets[i].AddFitness(inputs[(int)nearest * 3 + 1] * -0.1f);
                }
                else
                {
                    nets[i].AddFitness(inputs[(int)nearest * 3 + 1] * 0.1f);
                }
            }
        }

        static void FeedSecond(ref List<NN> nets, int populationSize, float[] inputs) //maitinimas 
        {
            List<float> perfectOutputs = new List<float>();

            for (int i = 0; i < 8; i++)
            {
                if (inputs[i] == 0)
                {
                    perfectOutputs.Add(i);
                }
            }

            for (int i = 0; i < populationSize; i++)    //kiekvienas NN
            {
                float output = (nets[i].FeedForward(inputs)[0] + 1f) * 4f;   //maitinami vieni duomenys

                float nearest = ClosestTo(perfectOutputs, output);

                //nets[i].AddFitness(1f - (nearest - Math.Abs(output)));    //pridedama kokybes tiksliam NN

                if ((int)nearest > 0 && (int)nearest < 6)
                {
                    if (inputs[(int)nearest - 1] == -1f || inputs[(int)nearest - 1] == -1f)
                    {
                        nets[i].AddFitness(0.2f);
                    }
                }

                nets[i].AddFitness((8f - nearest) / 16f);

            }
        }

        private static float[] GenerateClasses() //vienos dienos 8 pamoku duomenu sugeneravimas 
        {
            Random random = new Random(Guid.NewGuid().GetHashCode());   //sukuriamas random skaiciuotuvas su bet kokiu seed

            float determineOtherClasses = (float)random.NextDouble() * 10f;
            float determineSameClasses = (float)random.NextDouble() * 10f;
            int otherClasses = 0;
            int sameClasses = 0;

            float[] finalData = new float[] { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
            bool[] isTaken = new bool[] { false, false, false, false, false, false, false, false };

            //other classes

            if (determineOtherClasses <= 6.7f)
            {
                otherClasses++;
            }
            if (determineOtherClasses <= 7f)
            {
                otherClasses++;
            }
            if (determineOtherClasses <= 7.5f)
            {
                otherClasses++;
            }
            if (determineOtherClasses <= 9f)
            {
                otherClasses++;
            }
            if (determineOtherClasses <= 9.5f)
            {
                otherClasses++;
            }

            //same classes

            if (determineSameClasses <= 5f)
            {
                sameClasses++;
            }
            if (determineSameClasses <= 8f)
            {
                sameClasses++;
            }

            //TODO optimizuoti paieskas, labai daug speliojimu lieka, ir uzima laiko.

            for (int i = 0; i < otherClasses; i++)
            {
                int number = random.Next(1, 9);

                while (isTaken[number - 1])
                {
                    number = random.Next(1, 9);
                }

                finalData[number - 1] = 0f;
            }

            for (int i = 0; i < sameClasses; i++)
            {
                int number = random.Next(1, 9);

                while (isTaken[number - 1])
                {
                    number = random.Next(1, 9);
                }

                finalData[number - 1] = -1f;
            }

            return finalData;   //grazina sudeliota diena.
        }

        public static float[] TurnSecond(string[] toTurn, string index)
        {
            float[] res = new float[8];

            for (int i = 0; i < res.Length; i++)
            {
                if (toTurn[i] == "-" || toTurn[i] == "" || toTurn[i] == " ")
                {
                    res[i] = 1f;
                }
                else if (toTurn[i] == index)
                {
                    res[i] = -1f;
                }
                else
                {
                    res[i] = 0f;
                }
            }

            return res;
        }

        public static float[] TurnFirst(string[] toTurn, string index)
        {
            float[] res = new float[15];

            for (int i = 0; i < 15; i++)
            {
                res[i] = 0f;
            }

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if (toTurn[i * 8 + j] == index)
                    {
                        res[i * 3 + 1]++;
                    }
                    else if (toTurn[i * 8 + j] == "-" || toTurn[i * 8 + j] == " " || toTurn[i * 8 + j] == " ")
                    {
                        res[i * 3 + 2]++;
                    }
                    else
                    {
                        res[i * 3]++;
                    }
                }
            }

            return res;
        }

        public static float[] GenerateDays()
        {
            float[] day = new float[3];
            List<float> week = new List<float>();

            Random random = new Random(Guid.NewGuid().GetHashCode());
            int indicator;

            for (int i = 0; i < 5; i++)
            {
                indicator = random.Next(0, 8);
                day[0] = indicator;
                day[2] = 8 - indicator;

                if (indicator < 7)
                {
                    indicator = random.Next(0, indicator);
                }
                else
                {
                    indicator = 0;
                }

                day[1] = indicator;

                day[2] -= indicator;

                week.AddRange(day);
            }

            return week.ToArray();
        }

        static void TrainFirst(ref List<NN> nets, int[] layers)
        {
            TextWriter rf = new StreamWriter(@"weightValues1.txt", false, System.Text.Encoding.GetEncoding(1257));

            int populationSize = 500;    //kiek mokinti
            int generation = 0;
            int expectedGenerations = 100;   //kiek generationu kartoti
            int itterations = 500;

            InitNN(ref nets, ref populationSize, layers);   //aprasomas NN

            for (int i = 0; i < expectedGenerations; i++)   //kartoti kol pasieks norima kieki generationu
            {
                for (int j = 0; j < itterations; j++)   //maitinti po itterations kartu
                {
                    FeedFirst(ref nets, populationSize, GenerateDays());
                }

                Console.WriteLine("gen {0}, fitness {1}", i, nets[populationSize - 1].GetFitness());

                Update(ref generation, ref nets, ref populationSize, layers);   //ismetami blogi NN, pridedami nauji
            }

            nets[populationSize - 1].PrintWeights(rf);
            rf.Close();
        }

        public static float LaunchFirst(NN net, int[] layers, float[] chosen)
        {
            StreamReader sk = new StreamReader(@"weightValues1.txt", System.Text.Encoding.GetEncoding(1257));

            net.ReadWeights(sk);
            sk.Close();

            float output = ((net.FeedForward(chosen)[0]) + 1f) * 2.5f; //gaunamas outputas is neural network

            Console.WriteLine("First AI result: {0}", output);

            List<float> possibilities = new List<float>();  //galimos vietos ikelti pamoka

            for (int i = 0; i < 5; i++)
            {
                if (chosen[i * 3 + 2] == 0f || chosen[i * 3] + chosen[i * 3 + 1] == 8f)
                {
                    possibilities.Add(i);
                }

                if (chosen[i * 3 + 2] == 0f && chosen[i * 3] + chosen[i * 3 + 1] == 8f)
                {
                    possibilities.Add(i);
                }
            }

            float nearest = ClosestTo(possibilities, output);

            return nearest;
        }

        static void TrainSecond(ref List<NN> nets, int[] layers)
        {
            TextWriter rf = new StreamWriter(@"weightValues2.txt", false, System.Text.Encoding.GetEncoding(1257));

            int populationSize = 500;    //kiek mokinti
            int generation = 0;
            int expectedGenerations = 150;   //kiek generationu kartoti
            int itterations = 1000;

            InitNN(ref nets, ref populationSize, layers);   //aprasomas NN

            Update(ref generation, ref nets, ref populationSize, layers);

            for (int i = 0; i < expectedGenerations; i++)   //kartoti kol pasieks norima kieki generationu
            {
                for (int j = 0; j < itterations; j++)   //maitinti po itterations kartu
                {
                    FeedSecond(ref nets, populationSize, GenerateClasses());
                }

                Console.WriteLine("gen {0},   max fitness {1}", i, nets[populationSize - 1].GetFitness());

                Update(ref generation, ref nets, ref populationSize, layers);   //ismetami blogi NN, pridedami nauji
            }


            nets[populationSize - 1].PrintWeights(rf);
            rf.Close();
        }

        public static int LaunchSecond(NN net, int[] layers, ref string[] chosen, string reference)
        {
            StreamReader sk = new StreamReader(@"weightValues2.txt", System.Text.Encoding.GetEncoding(1257));

            net.ReadWeights(sk);
            sk.Close();

            const int dataCount = 8;                    //kiek pradiniu duomenu
            float[] primalData = new float[dataCount];  //naujas duomenu masyvas NN'ui

            for (int i = 0; i < dataCount; i++)
            {
                primalData[i] = 1f;
            }

            primalData = TurnSecond(chosen, reference);    //jau suzymetos klases ir langai paverciami i duomenis NN'ui

            float output = ((net.FeedForward(primalData)[0]) + 1f) * 4f;   //gaunamas outputas is neural network

            Console.WriteLine("Second AI result: {0}", output);

            List<float> possibilities = new List<float>();      //galimos vietos ikelti pamoka

            for (int j = 0; j < dataCount; j++)
            {
                if (chosen[j] == "-")
                {
                    possibilities.Add(j);     //surenkamos visos tuscios vietos
                }
            }

            float nearest = ClosestTo(possibilities, output);   //surandama artimiausia tuscia vieta pamokai

            return (int)nearest;
        }


        private static void TrainAll()
        {
            //-------------------primas NN-------------------
            // Tikslas: nustatyti optimaliausia diena pamokai iterpti
            // Duomenys: visos savaites klases, kai langas = 0, uzimta pamoka = 1, tokia pati pamoka = 0.5
            // Rezultatai: diena nuo 1 iki 5 i kuria reikia ideti pamoka
            //
            List<NN> firstNets = new List<NN>();  // NN tinklu sarasas
            int[] firstLayers = new int[] { 40, 20, 10, 1 };  //sluoksniu sistema

            NN firstNet = new NN(firstLayers);

            TrainFirst(ref firstNets, firstLayers);


            //-------------------antras NN-------------------
            // Tikslas: nustatyti optimaliausia langa pamokai iterpti
            // Duomenys: vienos dienos pamokos, kai langas = 0, uzimta pamoka = 1, tokia pati pamoka = 0.5
            // Rezultatai: langas nuo 1 iki 8 i kuri reikia ideti pamoka
            //
            List<NN> secondNets = new List<NN>();  // NN tinklu sarasas
            int[] secondLayers = new int[] { 8, 10, 10, 1 };  //sluoksniu sistema

            NN secondNet = new NN(secondLayers);

            TrainSecond(ref secondNets, secondLayers);
        }

        public static string[] ExpandClass(TTdata[] fullList, string teacher, bool[,] isOccupied)
        {
            //Tikslas: is turimu duomenu gauti string masyva su kiekviena klase/klasiu grupe, kurias reikia ideti
            // vienam mokytojui

            //TODO update. siek tiek pataisyta, dar truksta modifikavimo ir optimizavimo

            List<string> completeList = new List<string>();
            int fullAmount = 0;

            string singleClassUnit; //tai ka irasys i viena langeli. pvz 3A/3B
            string[] ClassUnit; //sarasas visu singleClassUnit

            string[] sameSubjectID;
            int n;

            bool[] notUsed = new bool[fullList.Length];

            const char separator = '/';

            for (int i = 0; i < fullList.Length; i++)
            {
                fullAmount += fullList[i].GetCount();
                notUsed[i] = true;  //visi priskiriami true
            }

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    notUsed[i * 8 + j] = false; //pagal mokytojo reikalavimus priskiriami false
                }
            }

            for (int i = 0; i < fullList.Length; i++)
            {
                ClassUnit = new string[fullList[i].GetCount()]; 
                sameSubjectID = new string[fullList.Length];

                n = 0;
                //notUsed[i] = false;

                for (int k = i; k < fullList.Length; k++)
                {
                    if (fullList[k].GetClassID() == fullList[i].GetClassID() && fullList[k].GetClass()[0] == fullList[i].GetClass()[0]
                        && fullList[k].GetTeacher() == fullList[i].GetTeacher() && fullList[k].GetCount() == fullList[i].GetCount()
                        && notUsed[k] && fullList[k].GetTeacher() == teacher)
                    {
                        sameSubjectID[n] = fullList[k].GetClass();  //isimenami ID identisku klasiu su ta pacia pamoka
                        n++;                                        //skaiciuojamas ju kiekis
                        notUsed[k] = false;
                    }
                }

                singleClassUnit = fullList[i].GetClass();

                for (int k = 0; k < n; k++)
                {
                    singleClassUnit += separator + fullList[k].GetClass();  //prie stringo pridedamos klases kurios mokosi kartu
                    // pvz 3G/3H/3F arba tiesiog 3A
                }

                for (int k = 0; k < fullList[i].GetCount(); k++)
                {
                    ClassUnit[k] = singleClassUnit; //sukuriamas stringas su klase 
                }

                completeList.AddRange(ClassUnit);   //prideda prie bendro saraso
            }

            return completeList.ToArray();
        }

        public static void BeginFeeding(ref string[,] timetableMatrica, int teacherIndex, string toInput, /**/ NN firstNet, NN secondNet, int[] firstLayers, int[] secondLayers)
        {
            //init

            //assumes timetableMatrica is not jagged array
            string[] byName = new string[timetableMatrica.GetLength(0)];
            float[] input;

            string[] dayClasses = new string[8];

            //set up data for feedforward
            for (int i = 0; i < timetableMatrica.Length; i++)
            {
                for (int j = 0; j < timetableMatrica.GetLength(i); j++)
                {
                    if (timetableMatrica[i, j] == toInput)
                    {
                        byName[i] = toInput;
                    }
                    else if (timetableMatrica[i, j] != " " && timetableMatrica[i, j] != "-" && timetableMatrica[i, j] != "")
                    {
                        byName[i] = timetableMatrica[i, j];
                    }
                    else
                    {
                        byName[i] = "-";
                    }
                }
            }

            input = TurnFirst(byName, toInput);
            float day = LaunchFirst(firstNet, firstLayers, input);

            for (int i = 0; i < 8; i++)
            {
                dayClasses[i] = byName[5 * (int)day + i];
            }

            int lesson = LaunchSecond(secondNet, secondLayers, ref dayClasses, toInput);

            timetableMatrica[teacherIndex, (int)day * 5 + lesson] = toInput;
        }

        //Sekanti dalis skirta ikelti i button_press ar kazka tokio,
        //cia todel kad negaliu tame darbe dabar keisti nieko, tai 
        //rasau cia

        public static void GenerateAllTimetables(TTdata[] fullList, bool[][,] preferedTimes)
        {
            //pakeiciami duomenys i klases, reikalingas irasyti i
            //lentele, t.y. klase arba kelios klases vienam mokytojui 
            //per viena pamoka. Po to pamokos surasomos mokytojams

            const char symbol = '-';

            List<string[,]> allTimetables = new List<string[,]>();  //pagrindine lentele
            List<string> classList = new List<string>();     //ka irasyti vienam mokytojui
            List<string> usedTeachers = new List<string>();     //jau surasyti mokytojai
            List<string> classes = new List<string>();     //visos esamos klases
            List<string> teacherList = new List<string>();     //visi mokytojai, gali ir kartotis

            for (int i = 0; i < fullList.Length; i++)
            {
                teacherList.Add(fullList[i].GetTeacher());

                //mokytoju sarasas (mokytojai kartojasi!)
                //to reikia kad skirtingus dalykus vienam mokytojui rodytu skirtingose eilutese tvarkarastyje
                //norint pakeisti, ideti if(teacherList.Contains(fullList[i].GetTeacher())) funkcija.
            }

            int[] firstLayers = new int[] { 40, 20, 10, 1 };
            int[] secondLayers = new int[] { 8, 10, 10, 1 };

            NN pickDay = new NN(firstLayers);
            NN pickLesson = new NN(secondLayers);

            for (int i = 0; i < fullList.Length; i++)   //sarasas visu klasiu, be pasikartojimu
            {
                if (!classes.Contains(fullList[i].GetTeacher()))
                {
                    classes.Add(fullList[i].GetClass());
                }
            }

            for (int i = 0; i < classes.Capacity; i++)  //visa tvarkarasti nustato i '-' simboli
            {
                for (int j = 0; j < teacherList.Capacity; j++)
                {
                    for (int k = 0; k < 40; k++)
                    {
                        allTimetables[i][j, k] = symbol.ToString();
                    }
                }
            }
            
            string[][,] complete = allTimetables.ToArray();     //sukuriamas specifinis trimatis masyvas
            int[] timetableFitness = new int[classes.Capacity]; //sukuriamas masyvas stebeti tvarkarasciu kokybei

            for (int u = 0; u < classes.Capacity; u++)
            {
                for (int i = 0; i < teacherList.Capacity; i++)
                {
                    if (!usedTeachers.Contains(fullList[i].GetTeacher()))
                    {
                        classList = ExpandClass(fullList, fullList[i].GetTeacher(), preferedTimes[i]).ToList();   //sukuriamas sarasas mokytojo pamokoms

                        usedTeachers.Add(fullList[i].GetTeacher()); //kad to paties mokytojo neidetu i kelias klases

                        for (int j = 0; j < classList.Capacity; j++)
                        {
                            BeginFeeding(ref complete[u], i, classList[j], pickDay, pickLesson, firstLayers, secondLayers); //ikeliaos pamokos mokytojams po viena
                        }
                    }
                }

                timetableFitness[u] = GenerateClassTimetable(fullList, classes[u], complete[u], teacherList);   //sudarytas vienas tvarkarastis
            }

            SortTimetables(ref complete, timetableFitness); //surusiuojami tvarkarasciai pagal ju kokybe, geriausi pirmieji

            //Prideti isvedima i dataGridView
            //Prideti esamo lango uzdaryma
            //Prideti naujo lango (perziurejimo lenteles) atidaryma

        }

        public static void SortTimetables(ref string[][,] complete, int[] fitness)  //Selection sort
        {
            int temp, smallest; //is tikro cia didziausias, bet pavadinima palikau
            string[,] tempT;
            for (int i = 0; i < fitness.Length - 1; i++)
            {
                smallest = i;
                for (int j = i + 1; j < fitness.Length; j++)
                {
                    if (fitness[j] > fitness[smallest])
                    {
                        smallest = j;
                    }
                }
                temp = fitness[smallest];
                fitness[smallest] = fitness[i];
                fitness[i] = temp;

                tempT = complete[smallest];
                complete[smallest] = complete[i];
                complete[i] = tempT;
            }
        }

        public static void ShiftByOne(ref List<string> list)
        {
            string last = list[list.Capacity];

            for (int i = 1; i < list.Capacity; i++)
            {
                list[i] = list[i - 1];
            }

            list[0] = last;
        }

        public static bool ContainsClass(string classList, string toFind)
        {
            if (toFind.Length == 2)
            {
                for (int i = 0; i < classList.Length - 1; i++)
                {
                    if (classList[i] == toFind[0] && classList[i + 1] == toFind[1])
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        public static int GenerateClassTimetable(TTdata[] fullList, string Class, string[,] fullTimetable, List<string> teacherList)
        {
            const int days = 5;
            const int lessons = 8;
            const char symbol = '-';
            const int maxPerLesson = 3;
            int fitness;

            string[,,] timetable = new string[days, lessons, maxPerLesson];

            for (int i = 0; i < days; i++)
            {
                for (int j = 0; j < lessons; j++)
                {
                    for (int k = 0; k < maxPerLesson; k++)
                    {
                        timetable[i, j, k] = symbol.ToString();
                    }
                }
            }

            for (int i = 0; i < teacherList.Capacity; i++)
            {
                for (int j = 0; j < days * lessons; j++)
                {
                    if (ContainsClass(fullTimetable[i, j], Class))
                    {
                        for (int k = 0; k < maxPerLesson; k++)
                        {
                            if(timetable[i / 8, i % 8, k] != symbol.ToString())
                            {
                                if (k != maxPerLesson - 1)
                                {
                                    timetable[i / 8, i % 8, k + 1] = fullList[i].GetSubject()[0] + fullList[i].GetSubject()[1].ToString();
                                }
                                else
                                {
                                    //patikra ar neperrasoma informacija
                                    // v ideti v
                                    //MessageBox.Show("ERROR: tvarkarascio duomenys perrasomi");
                                }
                            }
                        }
                    }
                }
            }

            fitness = TimetableFitness(timetable);

            return fitness;
        }

        public static int TimetableFitness(string[,,] timetable)
        {
            int fitness = 100;
            const char symbol = '-';

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    if (timetable[i, j, 0] == symbol.ToString())
                    {
                        fitness += -1;
                    }
                    if (j != 0 && timetable[i, j - 1, 0] == timetable[i, j, 0])
                    {
                        fitness += 2;
                    }
                }
            }

            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 8 - 2; j++)
                {
                    if (timetable[i, j, 0] != symbol.ToString() && timetable[i, j + 1, 0] == symbol.ToString() && timetable[i, j + 2, 0] != symbol.ToString())
                    {
                        fitness += -1;
                    }
                }
            }

            return fitness;
        }

        static void Main(string[] args)
        {
            //oops, nieko nera
        }
    }
}
