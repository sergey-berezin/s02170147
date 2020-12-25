using System;
using System.Linq;
using onnxModel;

namespace Main
{
    class Program
    {
        private static void PredictionHandler(PredictionPrint ResultPred)
        {
            Console.WriteLine(ResultPred.Prediction);
        }

        private static void OutputHandler(string outMessage)
        {
            Console.WriteLine(outMessage);
        }
        static void Main(string[] args)
        {
            var model = new OnnxModel("/Users/vladimirlisovoi/desktop/prak1/img" );

            model.EventResult += PredictionHandler;
            model.OutputEvent += OutputHandler;
            model.Work();
        }
    }
}