using System;
using System.CodeDom.Compiler;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Security;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using onnxModel;


namespace UI
{
    
    public partial class MainWindow : Window
    {
        private Model mdl = new Model();

        private ObservableCollection<Image> all_images;
        private ObservableCollection<ClassName> class_counts;
        
        private ObservableCollection<PredictionResult> all_results;
        private ObservableCollection<Image> selected_class_images;

        private ICollectionView list_box_predicted_labels_Updater;

        public static RoutedCommand Start = new RoutedCommand("Start", typeof(MainWindow));
        public static RoutedCommand Stop = new RoutedCommand("Stop", typeof(MainWindow));

        private bool isDirSelected = false;
        private bool isWorking = false;

        public string selected_dir;

        private ObservableCollection<string> db_stats;
        
        public void OutputHandler(PredictionResult current_result)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                all_results.Add(current_result);
                var current_class = from i in class_counts
                        where i.Class == current_result.Label
                        select i;
                if (current_class.Count() == 0)
                {
                    class_counts.Add(new ClassName(current_result.Label));
                }
                else
                {
                    current_class.First().Count++;
                    list_box_predicted_labels_Updater.Refresh();
                }

                var current_image = from i in all_images
                    where i.Path == current_result.Path
                    select i;
                current_image.First().Class = current_result.Label;


                
            }
            ));
        }
        private void OpenCommandHandler(object sender, ExecutedRoutedEventArgs e)
        {
            var dialog = new System.Windows.Forms.FolderBrowserDialog();
            System.Windows.Forms.DialogResult result = dialog.ShowDialog();
            if (result != System.Windows.Forms.DialogResult.OK)
            {
                return;
            }
            this.selected_dir = dialog.SelectedPath;
            this.isDirSelected = true;
        }
        private void StartCommandHandler(object sender, ExecutedRoutedEventArgs e)
        {
            this.isWorking = true;

            all_results.Clear();
            all_images.Clear();
            class_counts.Clear();
            ThreadPool.QueueUserWorkItem(new WaitCallback(param =>
            {
                foreach (string path in Directory.GetFiles(selected_dir, "*.jpg"))
                {
                    Dispatcher.BeginInvoke(new Action(() =>
                    {
                        all_images.Add(new Image(path));
                    }));
                }
            }));

            string model_path = "C:/Users/Владимир/prak1/s02170147/ImageProcessor/onnxModel/mnist-8.onnx";
            mdl = new Model(model_path, selected_dir);
            mdl.ResultEvent += OutputHandler;
            mdl.Work();
        }
        private void StopCommandHandler(object sender, ExecutedRoutedEventArgs e)
        {
            if (mdl != null)
            {
                mdl.Stop();
            }
            this.isWorking = false;
        }
        private void CanOpenCommandHanlder(object sender, CanExecuteRoutedEventArgs e)
        {
            e.CanExecute = !this.isWorking;
        }
        private void CanStartCommandHandler(object sender, CanExecuteRoutedEventArgs e)
        {
            e.CanExecute = !this.isWorking && this.isDirSelected;
        }
        private void CanStopCommandHandler(object sender, CanExecuteRoutedEventArgs e)
        {
            e.CanExecute = this.isWorking;
        }
        public MainWindow()
        {
            InitializeComponent();

            all_results = new ObservableCollection<PredictionResult>();
            all_images = new ObservableCollection<Image>();
  
            class_counts = new ObservableCollection<ClassName>();
            selected_class_images = new ObservableCollection<Image>();

            db_stats = new ObservableCollection<string>();

            Binding class_count = new Binding();
            class_count.Source = class_counts;
            list_box_predicted_labels.SetBinding(ItemsControl.ItemsSourceProperty, class_count);
            list_box_predicted_labels_Updater = CollectionViewSource.GetDefaultView(list_box_predicted_labels.ItemsSource);

            Binding for_selected_class = new Binding();
            for_selected_class.Source = selected_class_images;
            list_box_selected_images.SetBinding(ItemsControl.ItemsSourceProperty, for_selected_class);

            Binding for_db_stats = new Binding();
            for_db_stats.Source = db_stats;
            list_box_db_stats.SetBinding(ItemsControl.ItemsSourceProperty, for_db_stats);

            
        }
        private void list_box_predicted_labels_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            selected_class_images.Clear();

            ClassName selected_class = list_box_predicted_labels.SelectedItem as ClassName;
            if (selected_class == null) 
            {
                return;
            }
            foreach (var single_img in all_images)
            {
                if (single_img.Class == selected_class.Class)
                {
                    selected_class_images.Add(single_img);
                }
            }
        }

        private void Stats_Button_Click(object sender, RoutedEventArgs e)
        {
            db_stats.Clear();
            
            List<string> all_res = mdl.ShowDbStats();
            
            foreach (var item in all_res)
            {
                db_stats.Add(item);
            }
            list_box_db_stats.Items.Refresh();
            
        }

        private void Clear_Button_Click(object sender, RoutedEventArgs e)
        {
            mdl.ClearDB();
            db_stats.Clear();
            list_box_db_stats.Items.Refresh();
        }
    }

    public class ClassName : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        public string Class { get; set; }
        private int count;
        public int Count
        {
            get
            {
                return count;
            }
            set
            {
                count = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Count"));
            }
        }

        public ClassName(string class_name)
        {
            Class = class_name;
            Count = 1;
        }

        public override string ToString()
        {
            return Class + ": " + Count;
        }
    }
    
    public class Image : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        public string Path { get; set; }
        private string class_name;
        public string Class
        {
            get
            {
                return class_name;
            }
            set
            {
                class_name = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Class"));
            }
        }
        public BitmapImage Bitmap { get; set; }

        public Image(string path)
        {
            Path = path;
            Class = "";
            Bitmap = new BitmapImage(new Uri(path));
        }
    }
}