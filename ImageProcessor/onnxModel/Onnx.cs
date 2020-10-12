using System;
using System.IO;
using System.Linq;
using System.Threading;
using SixLabors.ImageSharp; // Из одноимённого пакета NuGet
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;

namespace onnxModel
{
    public class OnnxModel
    {
        private string path;
        private ConcurrentQueue<string> _filenames;
        InferenceSession session;
        private static readonly ManualResetEvent StopSignal = new ManualResetEvent(false);

        public OnnxModel(string Path = "Users/vladimirlisovoi/desktop/prak1/")
        {
            this.path = Path;
            this.session = new InferenceSession("/Users/vladimirlisovoi/desktop/prak1/ImageProcessor/onnxModel/mnist-8.onnx");
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

            var inputs = new List<NamedOnnxValue> {NamedOnnxValue.CreateFromTensor("Input3", input)};
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            var output = results.First().AsEnumerable<float>();
            var sum = output.Sum(x => (float)Math.Exp(x));
            var softmax = output.Select(x => (float)Math.Exp(x) / sum);

            var preds = softmax.Select((x, i) =>
                    new Tuple<string, float>(LabelMap.ClassLabels[i], x))
                    .OrderByDescending(x => x.Item2)
                    .Take(10);
            var prediction = "\n";
            foreach (var (label, confidence) in preds.ToList())
            {
                prediction += $"Label: {label}, confidence: {confidence}\n";
            }
            return prediction;
        
        }

        public void Work()
        {
            try
            {
                _filenames = new ConcurrentQueue<string>(Directory.GetFiles(path, "*.jpg"));
            }
            catch (DirectoryNotFoundException)
            {
                Console.WriteLine("No files");
                return;
            }
            Console.CancelKeyPress += (sender, eArgs) =>
            {
                StopSignal.Set();
                eArgs.Cancel = true;
            };
            var procNumb = Environment.ProcessorCount;
            var threads = new Thread[procNumb];
            for (var i = 0; i < procNumb; ++i)
            {
                Console.WriteLine($"Starting thread {i}");
                threads[i] = new Thread(Worker);
                threads[i].Start();
            }

            for (var i = 0; i < procNumb; ++i)
            {
                threads[i].Join();
            }

            Console.WriteLine("Success");
        }

        private void Worker()
        {
            while (_filenames.TryDequeue(out var name))
            {
                if (StopSignal.WaitOne(0))
                {
                    Console.WriteLine("Finishing thread");
                    return;
                }

                var prediction = FileRead(name);
                Console.WriteLine(name + prediction);

            }

            Console.WriteLine("Thread finished");
        }

    }


    static class LabelMap
    {
        public static readonly string[] ClassLabels = new[]
        {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        };
    }

} 
