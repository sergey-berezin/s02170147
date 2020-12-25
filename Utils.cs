using Microsoft.EntityFrameworkCore;


namespace onnxModel
{
    public class Result
    {
        public int ResultId { get; set; }
        public byte[] Hash { get; set; }
        public string Path { get; set; }
        public string Label { get; set; }
        public double Confidence { get; set; }
        public int CountReffered { get; set; }
        public virtual ImgDetail Detail { get; set; }
    };
    public class ImgDetail
    {
        public int ImgDetailId { get; set; }
        public byte[] RawImg { get; set; }
    };

    class MyResultContext : DbContext
    {
        public DbSet<Result> Results { get; set; }
        public DbSet<ImgDetail> ImgDetails { get; set; }

        

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder) =>
        optionsBuilder.UseSqlite(@"Data Source=..\..\..\..\library.db");

        public void Clear()
        {
            lock (this)
            {
                Results.RemoveRange(Results);
                ImgDetails.RemoveRange(ImgDetails);
                SaveChanges();
            }
        }
    }
}