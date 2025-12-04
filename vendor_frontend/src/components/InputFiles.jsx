import { useState } from "react";
import axios from "axios";
import { Button } from "../components/ui/button";

const InputFiles = () => {
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) setFiles((prev) => prev.concat(selected));
  };

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!files.length) return;

    const token = localStorage.getItem("token");
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setIsUploading(true);

    try {
      await axios.post("/vendor-api/files", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`
        }
      });
    } finally {
      setIsUploading(false);
      setFiles([]);
    }
  };

  return (
    <div className="flex flex-col items-center min-w-96 pt-6">

      <div className="flex justify-center gap-5 w-full">
        <input
          type="file"
          className="hidden"
          id="file-input"
          onChange={handleFileChange}
        />

        <label htmlFor="file-input">
          <Button asChild>
            <span>Choose File</span>
          </Button>
        </label>

        <Button onClick={handleUpload}>
          Upload
        </Button>
      </div>

      <div className="flex flex-col flex-wrap pt-12">
        {files.map((file, index) => (
          <span key={index} className="text-sm text-muted-foreground">
            {file.name}
          </span>
        ))}
      </div>

      {isUploading && (
        <span className="pt-4 text-sm text-muted-foreground">
          Uploading...
        </span>
      )}
    </div>
  );
};

export default InputFiles;
