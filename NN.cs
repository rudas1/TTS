using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetwork
{
    public class NN
    {
        private int[] layers;
        private float[][] neurons;
        private float[][][] weights;
        private float fitness;

        private Random rnd;

        public NN(int[] layers) // Constructor, sukuria savo nauja matrica
        {
            this.layers = new int[layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                this.layers[i] = layers[i];
            }

            rnd = new Random(); //random rodiklis

            InitNeurons();  //sukuriami neuronai
            InitWeights();  //sukuriami svoriai

        } // NN()

        public NN(NN copyNetwork)   // Constructor, perkopijuoja kita NN
        {
            this.layers = new int[copyNetwork.layers.Length];

            for (int i = 0; i < copyNetwork.layers.Length; i++)
            {
                this.layers[i] = copyNetwork.layers[i]; //kopijuojamas kiekvienas sluoksnis kaip naujas
            }

            rnd = new Random(); //random rodiklis

            InitNeurons();  //aprasomi neuronai
            InitWeights();  //aprasomi svoriai

            CopyWeights(copyNetwork.weights);   //svoriu reiksmes perkopijuojamos is kito NN

        } // NN()

        private void CopyWeights(float[][][] copyWeights)   //is kito NN pasiima visa svoriu kopija
        {
            for (int i = 0; i < weights.Length; i++)    //sluoksnis
            {
                for (int j = 0; j < weights[i].Length; j++) //neuronas
                {
                    for (int k = 0; k < weights[i][j].Length; k++)  //praeito sluoksnio neuronas
                    {
                        weights[i][j][k] = copyWeights[i][j][k];    //priskiaramos kito NN svoriu reiksmes
                    } //k
                } //j
            } //i
        } // CopyWeights()

        private void InitNeurons()  //neuronu apsirasymas
        {
            List<float[]> neuronsList = new List<float[]>();    //naujas listas > jagged array

            for (int i = 0; i < layers.Length; i++)
            {
                neuronsList.Add(new float[layers[i] + 1]);  //sukuriamas vienas layer
                neuronsList[i][layers[i]] = 1;
            }

            neurons = neuronsList.ToArray();    //paverciamas i array
        } //Init Neurons()

        private void InitWeights()  //sukuriamas svoriu tinklas
        {
            List<float[][]> weightsList = new List<float[][]>();    //bendra svoriu matrica su listu

            for (int i = 1; i < layers.Length; i++) //kiekvienas neuronas su kiekvienu praeito sluoksnio neuronu
            {
                List<float[]> layerWeightsList = new List<float[]>();   //bendra vieno sluoksnio svoriu matrica

                int neuronsInPreviousLayer = layers[i - 1] + 1; //informacija apie praeita neuronu sluoksni

                for (int j = 0; j < neurons[i].Length; j++) //kiekvienas neuronas esamam sluoksnyje
                {
                    float[] neuronWeights = new float[neuronsInPreviousLayer]; //vienas neuronas su visais praeito sluoksnio neuronais

                    for (int k = 0; k < neuronsInPreviousLayer; k++)    //kiekvienas neuronas praeitame sluoksnyje
                    {
                        neuronWeights[k] = (float)rnd.NextDouble();   //random nuo -0.5 iki 0.5
                    }

                    layerWeightsList.Add(neuronWeights);    //prie sluoksnio neuronu su svoriais pridedamas dar vienas
                }

                weightsList.Add(layerWeightsList.ToArray());//sluoksnis pridedamas prie bendro svoriu saraso
            }

            weights = weightsList.ToArray();    //priskiriami duomenys
        } // InitWeights()

        public float[] FeedForward(float[] inputs)  //duomenys siunciami pro neuron network
        {
            for (int i = 0; i < inputs.Length; i++) //i duomenu neuronus ikeliami duomenys
            {
                neurons[0][i] = inputs[i];
            }

            for (int i = 1; i < layers.Length; i++) //praeinama pro kiekviena sluoksni
            {
                for (int j = 0; j < neurons[i].Length; j++) //praeinama pro kiekviena sluoksnio neurona
                {
                    float value = 0f;

                    //Console.WriteLine(i - 1);
                    for (int k = 0; k < neurons[i - 1].Length; k++) //praeinama pro kiekviena svori sujungta su neuronu
                    {
                        value += weights[i - 1][j][k] * neurons[i - 1][k];  //pridedama praeito sluoksnio neurono vertes ir su juo sujungto svorio sandauga
                    }

                    neurons[i][j] = (1f / (1f + (float)Math.Exp(-value))) - 0.5f;    //ivykdoma aktyvacija - hyperbolic tangent (0 - 1)
                }
            }

            return neurons[neurons.Length - 1]; //grazinamas output sluoksnis
        } // FeedForward()

        public void Mutate()    //atsitiktinis svoriu koregavimas
        {
            for (int i = 0; i < weights.Length; i++)    //sluoksnis
            {
                for (int j = 0; j < weights[i].Length; j++) //neuronas
                {
                    for (int k = 0; k < weights[i][j].Length; k++)  //sujungtas svoris
                    {
                        float weight = weights[i][j][k];    //perkopijuojama i atskira float

                        float randomNum = (float)(rnd.NextDouble() * 100f);

                        if(randomNum <= 2f)
                        {
                            //pirma variacija
                            //apverciama svorio reiksme
                            weight *= -1f;
                        }
                        else if (randomNum <= 4f)
                        {
                            //antra variacija
                            //nauja bet kokia svorio reiksme
                            weight = (float)((rnd.NextDouble() - 0.5f) * 4);
                        }
                        else if (randomNum <= 6f)
                        {
                            //trecia variacija
                            //reiksme padidinama nuo 0% iki 100%
                            float factor = (float)(rnd.NextDouble() + 1f);
                            weight *= factor;
                        }
                        else if (randomNum <= 8f)
                        {
                            //ketvirta variacija
                            //reiksme sumazinama nuo 0% iki 100%
                            float factor = (float)(rnd.NextDouble());
                            weight *= factor;
                        }

                        weights[i][j][k] = weight;  //atgal priskirama reiksme svoriui
                    } // k
                } // j
            } // i
        } // Mutate()

        public void PrintWeights(TextWriter rf)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        rf.WriteLine(weights[i][j][k]);
                        rf.Flush();
                    }
                }
            }
        }

        public void ReadWeights(TextReader sk)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        weights[i][j][k] = Single.Parse(sk.ReadLine().ToString());
                        //Console.Write(weights[i][j][k]);
                        //Console.Write(" ");
                    }
                    //Console.WriteLine();
                }
                //Console.WriteLine();
            }
        }

        public void AddFitness(float fit)   //pridedamas kokybes lygis
        {
            fitness += fit;
        }

        public void SetFitness(float fit)   //nustatomas kokybes lygis
        {
            fitness = fit;
        }

        public float GetFitness()           //gaunamas kokybes lygis
        {
            return fitness;
        }

        public int CompareTo(NN other)  //skirtas ivertinti ar NN yra geresnis uz kita
        {
            if (other == null) return 1;

            if (fitness > other.fitness) return 1;

            else if (fitness <= other.fitness) return -1;

            else return 0;

            //reiksme 1: esamas NN yra geresnis arba kitas NN neaprasytas
            //reiksme 0: klaida
            //reiksme -1: esamas NN yra blogesnis uz kita NN
        } // CompareTo()
    }
}
