using System;
using Microsoft.EntityFrameworkCore.Migrations;

namespace onnxModel.Migrations
{
    public partial class FirstMigration : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "ImgDetails",
                columns: table => new
                {
                    ImgDetailId = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    RawImg = table.Column<byte[]>(type: "BLOB", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ImgDetails", x => x.ImgDetailId);
                });

            migrationBuilder.CreateTable(
                name: "Results",
                columns: table => new
                {
                    ResultId = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    Hash = table.Column<byte[]>(type: "BLOB", nullable: true),
                    Path = table.Column<string>(type: "TEXT", nullable: true),
                    Label = table.Column<string>(type: "TEXT", nullable: true),
                    Confidence = table.Column<double>(type: "REAL", nullable: false),
                    CountReffered = table.Column<int>(type: "INTEGER", nullable: false),
                    DetailImgDetailId = table.Column<int>(type: "INTEGER", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Results", x => x.ResultId);
                    table.ForeignKey(
                        name: "FK_Results_ImgDetails_DetailImgDetailId",
                        column: x => x.DetailImgDetailId,
                        principalTable: "ImgDetails",
                        principalColumn: "ImgDetailId",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Results_DetailImgDetailId",
                table: "Results",
                column: "DetailImgDetailId");
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Results");

            migrationBuilder.DropTable(
                name: "ImgDetails");
        }
    }
}
