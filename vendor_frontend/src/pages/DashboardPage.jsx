import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import {
  Upload,
  Image as ImageIcon,
  FileText,
  X,
  Plus,
  LogOut,
  Store,
  Package,
  IndianRupee,
  Tag,
  Sparkles
} from "lucide-react";

const DashboardPage = () => {
  const [vendorInfo, setVendorInfo] = useState(null);
  const [productData, setProductData] = useState({
    name: "",
    category: "",
    subcategory: "",
    price: "",
    unit: "item",
    description: "",
    colors: [""],
    sizes: [""],
    materials: "",
    careInstructions: "",
    stock: "",
    brand: "",
    dimensions: ""
  });
  const [images, setImages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [imagePreviews, setImagePreviews] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    fetchVendorInfo();
  }, []);

  const fetchVendorInfo = async () => {
    try {
      const token = localStorage.getItem("token");
      const response = await fetch("/vendor-api/users/me", {
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setVendorInfo(data);
      }
    } catch (err) {
      console.error("Error fetching vendor info:", err);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  const handleImageChange = (e) => {
    const files = Array.from(e.target.files);
    setImages(prev => [...prev, ...files]);
    
    // Create previews
    files.forEach(file => {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreviews(prev => [...prev, reader.result]);
      };
      reader.readAsDataURL(file);
    });
  };

  const handleDocumentChange = (e) => {
    const files = Array.from(e.target.files);
    setDocuments(prev => [...prev, ...files]);
  };

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
    setImagePreviews(prev => prev.filter((_, i) => i !== index));
  };

  const removeDocument = (index) => {
    setDocuments(prev => prev.filter((_, i) => i !== index));
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setProductData(prev => ({ ...prev, [name]: value }));
  };

  const handleArrayInput = (field, index, value) => {
    setProductData(prev => ({
      ...prev,
      [field]: prev[field].map((item, i) => i === index ? value : item)
    }));
  };

  const addArrayItem = (field) => {
    setProductData(prev => ({
      ...prev,
      [field]: [...prev[field], ""]
    }));
  };

  const removeArrayItem = (field, index) => {
    setProductData(prev => ({
      ...prev,
      [field]: prev[field].filter((_, i) => i !== index)
    }));
  };

  const handleSubmit = async () => {
    setMessage("");

    // Validation - mandatory fields
    if (!productData.name) {
      setMessage("Product name is required");
      return;
    }
    if (!productData.category) {
      setMessage("Category is required");
      return;
    }
    if (!productData.subcategory) {
      setMessage("Subcategory is required");
      return;
    }
    if (!productData.price || parseFloat(productData.price) <= 0) {
      setMessage("Price is required and must be greater than 0");
      return;
    }
    if (!productData.stock || parseInt(productData.stock) < 0) {
      setMessage("Stock quantity is required");
      return;
    }
    if (!images.length) {
      setMessage("Please upload at least one product image");
      return;
    }

    setUploading(true);

    try {
      const token = localStorage.getItem("token");

      // Prepare FormData
      const formData = new FormData();

      // Add all files
      images.forEach(img => formData.append("files", img));
      documents.forEach(doc => formData.append("files", doc));

      // Prepare product data
      const productPayload = {
        name: productData.name,
        category: productData.category,
        subcategory: productData.subcategory,
        price: parseFloat(productData.price),
        description: productData.description || "",
        colors: productData.colors.filter(c => c.trim() !== ""),
        sizes: productData.sizes.filter(s => s.trim() !== ""),
        materials: productData.materials || "",
        care_instructions: productData.careInstructions || "",
        stock: parseInt(productData.stock) || 0,
        dimensions: productData.dimensions || "",
        brand: productData.brand || "",
        unit: productData.unit || "item"
      };
      
      // Add product data as JSON string
      formData.append("product_data", JSON.stringify(productPayload));
      
      const uploadResponse = await fetch("/vendor-api/files/", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`
        },
        body: formData
      });
      
      if (uploadResponse.ok) {
        const result = await uploadResponse.json();
        setMessage("Product uploaded successfully!");
        console.log("Product created:", result.product);
        
        // Reset form
        setProductData({
          name: "",
          category: "",
          subcategory: "",
          price: "",
          unit: "item",
          description: "",
          colors: [""],
          sizes: [""],
          materials: "",
          careInstructions: "",
          stock: "",
          brand: "",
          dimensions: "",
        });
        setImages([]);
        setDocuments([]);
        setImagePreviews([]);
      } else {
        const error = await uploadResponse.json();
        setMessage(`Upload failed: ${error.detail || "Please try again."}`);
      }
    } catch (err) {
      console.error("Upload error:", err);
      setMessage("Error uploading product. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-linear-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <Store className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">
                {vendorInfo?.business_name || "Vendor Dashboard"}
              </h1>
              <p className="text-sm text-slate-400">@{vendorInfo?.username}</p>
            </div>
          </div>
          <Button
            onClick={handleLogout}
            className="bg-slate-800 hover:bg-slate-700 text-white"
          >
            <LogOut className="w-4 h-4 mr-2" />
            Logout
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Welcome Banner */}
        <div className="bg-linear-to-r from-blue-600 to-purple-600 rounded-2xl p-8 mb-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl"></div>
          <div className="relative">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-6 h-6 text-yellow-300" />
              <span className="text-sm font-medium text-blue-100">Welcome back!</span>
            </div>
            <h2 className="text-3xl font-bold text-white mb-2">
              Upload Your Products
            </h2>
            <p className="text-blue-100">
              Share your amazing products with customers. Add detailed information and high-quality images.
            </p>
          </div>
        </div>

        {/* Upload Form */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Product Details */}
          <div className="lg:col-span-2 space-y-6">
            {/* Basic Information */}
            <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                <Package className="w-5 h-5 text-blue-400" />
                Basic Information
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Product Name *
                  </label>
                  <input
                    name="name"
                    value={productData.name}
                    onChange={handleInputChange}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    placeholder="e.g., Cotton Summer T-Shirt"
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Category <span className="text-red-400">*</span>
                    </label>
                    <select
                      name="category"
                      value={productData.category}
                      onChange={handleInputChange}
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    >
                      <option value="">Select category</option>
                      <option value="mens_clothing">Men's Clothing</option>
                      <option value="womens_clothing">Women's Clothing</option>
                      <option value="kids_clothing">Kids Clothing</option>
                      <option value="accessories">Accessories</option>
                      <option value="footwear">Footwear</option>
                      <option value="textiles">Textiles & Fabrics</option>
                      <option value="home_decor">Home Decor</option>
                      <option value="other">Other</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Subcategory <span className="text-red-400">*</span>
                    </label>
                    {productData.category === "other" ? (
                      <input
                        name="subcategory"
                        value={productData.subcategory}
                        onChange={handleInputChange}
                        className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                        placeholder="Enter custom subcategory"
                      />
                    ) : (
                      <input
                        name="subcategory"
                        value={productData.subcategory}
                        onChange={handleInputChange}
                        className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                        placeholder="e.g., Blouse, Shirt, Dress"
                      />
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Price <span className="text-red-400">*</span>
                    </label>
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">₹</span>
                        <input
                          name="price"
                          type="number"
                          step="0.01"
                          min="0.01"
                          required
                          value={productData.price}
                          onChange={handleInputChange}
                          className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-8 pr-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                          placeholder="Enter price"
                        />
                      </div>
                      <select
                        name="unit"
                        value={productData.unit}
                        onChange={handleInputChange}
                        className="bg-slate-800/50 border border-slate-700 rounded-lg px-3 py-2.5 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition min-w-[120px]"
                      >
                        <option value="item">per item</option>
                        <option value="piece">per piece</option>
                        <option value="kg">per kg</option>
                        <option value="g">per 100g</option>
                        <option value="litre">per litre</option>
                        <option value="ml">per 500ml</option>
                        <option value="dozen">per dozen</option>
                        <option value="pair">per pair</option>
                        <option value="set">per set</option>
                        <option value="meter">per meter</option>
                        <option value="sqft">per sq.ft</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Stock Quantity <span className="text-red-400">*</span>
                  </label>
                  <input
                    name="stock"
                    type="number"
                    min="0"
                    required
                    value={productData.stock}
                    onChange={handleInputChange}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    placeholder="Available quantity"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Description
                  </label>
                  <textarea
                    name="description"
                    value={productData.description}
                    onChange={handleInputChange}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition resize-none"
                    rows="4"
                    placeholder="Describe your product in detail..."
                  />
                </div>
              </div>
            </div>

            {/* Product Specifications */}
            <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                <Tag className="w-5 h-5 text-purple-400" />
                Product Specifications
              </h3>

              <div className="space-y-4">
                {/* Colors */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Available Colors
                  </label>
                  {productData.colors.map((color, index) => (
                    <div key={index} className="flex gap-2 mb-2">
                      <input
                        value={color}
                        onChange={(e) => handleArrayInput("colors", index, e.target.value)}
                        className="flex-1 bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="e.g., Navy Blue, Red"
                      />
                      {productData.colors.length > 1 && (
                        <Button
                          onClick={() => removeArrayItem("colors", index)}
                          className="bg-slate-800 hover:bg-slate-700"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                  <Button
                    onClick={() => addArrayItem("colors")}
                    className="bg-slate-800 hover:bg-slate-700 text-white mt-2"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Color
                  </Button>
                </div>

                {/* Sizes - Only show for clothing categories */}
                {["mens_clothing", "womens_clothing", "kids_clothing", "footwear"].includes(productData.category) && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Available Sizes
                  </label>
                  {productData.sizes.map((size, index) => (
                    <div key={index} className="flex gap-2 mb-2">
                      <input
                        value={size}
                        onChange={(e) => handleArrayInput("sizes", index, e.target.value)}
                        className="flex-1 bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="e.g., S, M, L, XL"
                      />
                      {productData.sizes.length > 1 && (
                        <Button
                          onClick={() => removeArrayItem("sizes", index)}
                          className="bg-slate-800 hover:bg-slate-700"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                  <Button
                    onClick={() => addArrayItem("sizes")}
                    className="bg-slate-800 hover:bg-slate-700 text-white mt-2"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Size
                  </Button>
                </div>
                )}

                {/* Materials */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Materials
                  </label>
                  <input
                    name="materials"
                    value={productData.materials}
                    onChange={handleInputChange}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., 100% Cotton, Polyester blend"
                  />
                </div>

                {/* Care Instructions */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Care Instructions
                  </label>
                  <textarea
                    name="careInstructions"
                    value={productData.careInstructions}
                    onChange={handleInputChange}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                    rows="3"
                    placeholder="e.g., Machine wash cold, tumble dry low"
                  />
                </div>

                {/* Brand */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Brand
                  </label>
                  <input
                    name="brand"
                    value={productData.brand}
                    onChange={handleInputChange}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., UrbanWrap, Nike, Adidas"
                  />
                </div>

                {/* Dimensions */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Dimensions (Optional)
                  </label>
                  <input
                    name="dimensions"
                    value={productData.dimensions}
                    onChange={handleInputChange}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., 34 x 26 x 3 cm"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Media Upload */}
          <div className="space-y-6">
            {/* Image Upload */}
            <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <ImageIcon className="w-5 h-5 text-green-400" />
                Product Images *
              </h3>

              <div className="space-y-4">
                <label className="block">
                  <div className="border-2 border-dashed border-slate-700 rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition">
                    <Upload className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                    <p className="text-sm text-slate-400">Click to upload images</p>
                    <p className="text-xs text-slate-600 mt-1">PNG, JPG up to 10MB</p>
                  </div>
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                </label>

                {/* Image Previews */}
                {imagePreviews.length > 0 && (
                  <div className="grid grid-cols-2 gap-3">
                    {imagePreviews.map((preview, index) => (
                      <div key={index} className="relative group">
                        <img
                          src={preview}
                          alt={`Preview ${index + 1}`}
                          className="w-full h-32 object-cover rounded-lg border border-slate-700"
                        />
                        <button
                          onClick={() => removeImage(index)}
                          className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Document Upload */}
            <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
              <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5 text-orange-400" />
                Documents (Optional)
              </h3>

              <div className="space-y-4">
                <label className="block">
                  <div className="border-2 border-dashed border-slate-700 rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition">
                    <Upload className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                    <p className="text-sm text-slate-400">Upload certificates, specs</p>
                    <p className="text-xs text-slate-600 mt-1">PDF, DOCX up to 5MB</p>
                  </div>
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.doc,.docx"
                    onChange={handleDocumentChange}
                    className="hidden"
                  />
                </label>

                {documents.length > 0 && (
                  <div className="space-y-2">
                    {documents.map((doc, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between bg-slate-800/50 rounded-lg p-3"
                      >
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-orange-400" />
                          <span className="text-sm text-slate-300 truncate">
                            {doc.name}
                          </span>
                        </div>
                        <button
                          onClick={() => removeDocument(index)}
                          className="text-red-400 hover:text-red-300"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Submit Button */}
            <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
              {message && (
                <div className={`mb-4 p-4 rounded-lg ${
                  message.includes("success") 
                    ? "bg-green-500/10 border border-green-500/20 text-green-400" 
                    : "bg-red-500/10 border border-red-500/20 text-red-400"
                }`}>
                  {message}
                </div>
              )}

              <Button
                onClick={handleSubmit}
                disabled={uploading}
                className="w-full bg-linear-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl disabled:opacity-50"
              >
                {uploading ? "Uploading..." : "Upload Product"}
              </Button>

              <p className="text-xs text-slate-500 text-center mt-3">
                * Required fields
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;