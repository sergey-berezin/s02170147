using System;
using System.IO;
using System.Linq;
using System.Threading;
using SixLabors.ImageSharp; // Из одноимённого пакета NuGet
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace onnxModel
{
    public class OnnxModel
    {

        string path, inputname;
        InferenceSession session;

        public string Path
        {
            get {return path;}
            set {path = value;}
        }
        public string InputName
        {
            get {return inputname;}
            set {inputname = value;}
        }
        public OnnxModel(string Path="mnist-8.onnx", string InputName="Input3")
        {
            this.path = Path;
            this.inputname = InputName;
            this.session = new InferenceSession(path);
        }

        public object FileRead(string ImagePath)
        {
            Image<Rgb24> image = Image.Load<Rgb24>(ImagePath);

            const int TargetWidth = 28;
            const int TargetHeight = 28;

            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetWidth, TargetHeight),
                    Mode = ResizeMode.Crop
                });
                x.Grayscale();
            });

            var input = new DenseTensor<float>(new[] { 1, 1, TargetHeight, TargetWidth });
            for (int y = 0; y < TargetHeight; y++)         
                for (int x = 0; x < TargetWidth; x++)
                    input[0, 0, y, x] = image[x,y].R / 255f;

            var inputs = new List<NamedOnnxValue>  
            { 
                NamedOnnxValue.CreateFromTensor("Input3", input) 
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            var sum = output.Sum(x => (float)Math.Exp(x));
            var softmax = output.Select(x => (float)Math.Exp(x) / sum);

            float maxsoftmax = softmax.Max();
            var l = softmax.ToList();
            int maxsoftmaxindex = l.IndexOf(maxsoftmax);

            return (maxsoftmaxindex, maxsoftmax);

        }

    }

    public delegate void OutputHandler(object sender, params object[] args);
    public class AsyncProcessor
    {
        static CancellationTokenSource cancel = new CancellationTokenSource();

        OnnxModel model;
        public event OutputHandler DataEvent;

        public AsyncProcessor(OnnxModel model)
        {
            this.model = model;
            Console.CancelKeyPress += new ConsoleCancelEventHandler((s, args)=>{args.Cancel = true; cancel.Cancel();});
        }

        public void Worker(string DirectoryPath)
        {

            cancel = new CancellationTokenSource();

            string[] Files = Directory.GetFiles(DirectoryPath);

            Thread[] threads = new Thread[Environment.ProcessorCount];

            int processed = -1;

            for (int i = 0; i < Environment.ProcessorCount; i++)
            {
                threads[i] = new Thread(()=>
                {
                    int Numb;

                    while(!cancel.Token.IsCancellationRequested)
                    {

                        Numb = Interlocked.Increment(ref processed);
                        if(Numb >= Files.Count())
                            break;
                        else
                        {
                            object output = model.FileRead(Files[Numb]);
                            DataEvent?.Invoke(this, Files[Numb], output);

                        }
                    }


                });
                threads[i].Start();
            }

             for (int i = 0; i < Environment.ProcessorCount; i++)
            {
                threads[i].Join();
            }

        }

    }

} 
