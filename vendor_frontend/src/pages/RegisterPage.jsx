import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Store, Mail, Phone, MapPin, Building2, User, Lock } from "lucide-react";

const RegisterPage = () => {
  const [formData, setFormData] = useState({
    username: "",
    password: "",
    confirmPassword: "",
    businessName: "",
    email: "",
    phone: "",
    address: "",
    businessType: "",
    description: ""
  });
  const [msg, setMsg] = useState("");
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.username || formData.username.length < 3) {
      newErrors.username = "Username must be at least 3 characters";
    }
    if (!formData.password || formData.password.length < 6) {
      newErrors.password = "Password must be at least 6 characters";
    }
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
    }
    if (!formData.businessName) {
      newErrors.businessName = "Business name is required";
    }
    if (!formData.email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = "Valid email is required";
    }
    if (!formData.phone || !/^\+?[\d\s-()]+$/.test(formData.phone)) {
      newErrors.phone = "Valid phone number is required";
    }
    if (!formData.businessType) {
      newErrors.businessType = "Please select a business type";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: "" }));
    }
  };

  const submit = async (e) => {
    e.preventDefault();
    setMsg("");
    
    if (!validateForm()) return;

    try {
      // For now, we'll just use username and password for the API
      // You'll need to update your backend to accept additional fields
      const response = await fetch("http://localhost:8000/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: formData.username,
          password: formData.password,
          business_name: formData.businessName,
          email: formData.email,
          phone: formData.phone,
          address: formData.address,
          business_type: formData.businessType,
          description: formData.description
        })
      });

      if (response.ok) {
        setMsg("Registration successful! You can now log in.");
        setTimeout(() => window.location.href = "/login", 2000);
      } else {
        setMsg("Username or email already exists.");
      }
    } catch (err) {
      setMsg("Registration failed. Please try again.");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 mb-4">
            <Store className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">Vendor Registration</h1>
          <p className="text-slate-400">Join our marketplace and showcase your products</p>
        </div>

        {/* Form Card */}
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 shadow-2xl">
          <form onSubmit={submit} className="space-y-6">
            {/* Account Credentials Section */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <User className="w-5 h-5 text-blue-400" />
                Account Credentials
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Username *
                  </label>
                  <input
                    name="username"
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    placeholder="Choose a username"
                    value={formData.username}
                    onChange={handleChange}
                  />
                  {errors.username && <p className="text-red-400 text-sm mt-1">{errors.username}</p>}
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Email *
                  </label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                    <input
                      name="email"
                      type="email"
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-11 pr-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                      placeholder="vendor@example.com"
                      value={formData.email}
                      onChange={handleChange}
                    />
                  </div>
                  {errors.email && <p className="text-red-400 text-sm mt-1">{errors.email}</p>}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Password *
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                    <input
                      name="password"
                      type="password"
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-11 pr-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                      placeholder="••••••••"
                      value={formData.password}
                      onChange={handleChange}
                    />
                  </div>
                  {errors.password && <p className="text-red-400 text-sm mt-1">{errors.password}</p>}
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Confirm Password *
                  </label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                    <input
                      name="confirmPassword"
                      type="password"
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-11 pr-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                      placeholder="••••••••"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                    />
                  </div>
                  {errors.confirmPassword && <p className="text-red-400 text-sm mt-1">{errors.confirmPassword}</p>}
                </div>
              </div>
            </div>

            {/* Business Information Section */}
            <div className="space-y-4 pt-4 border-t border-slate-800">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <Building2 className="w-5 h-5 text-purple-400" />
                Business Information
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Business Name *
                  </label>
                  <input
                    name="businessName"
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    placeholder="Your Business Name"
                    value={formData.businessName}
                    onChange={handleChange}
                  />
                  {errors.businessName && <p className="text-red-400 text-sm mt-1">{errors.businessName}</p>}
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Business Type *
                  </label>
                  <select
                    name="businessType"
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    value={formData.businessType}
                    onChange={handleChange}
                  >
                    <option value="">Select type</option>
                    <option value="clothing">Clothing & Apparel</option>
                    <option value="textiles">Textiles & Fabrics</option>
                    <option value="accessories">Accessories</option>
                    <option value="footwear">Footwear</option>
                    <option value="jewelry">Jewelry</option>
                    <option value="home_decor">Home Decor</option>
                    <option value="other">Other</option>
                  </select>
                  {errors.businessType && <p className="text-red-400 text-sm mt-1">{errors.businessType}</p>}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Phone Number *
                </label>
                <div className="relative">
                  <Phone className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                  <input
                    name="phone"
                    type="tel"
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-11 pr-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                    placeholder="+1 (555) 123-4567"
                    value={formData.phone}
                    onChange={handleChange}
                  />
                </div>
                {errors.phone && <p className="text-red-400 text-sm mt-1">{errors.phone}</p>}
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Business Address
                </label>
                <div className="relative">
                  <MapPin className="absolute left-3 top-3 w-5 h-5 text-slate-500" />
                  <textarea
                    name="address"
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-11 pr-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition resize-none"
                    placeholder="Street address, city, state, zip"
                    rows="2"
                    value={formData.address}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Business Description
                </label>
                <textarea
                  name="description"
                  className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition resize-none"
                  placeholder="Tell us about your business and products..."
                  rows="3"
                  value={formData.description}
                  onChange={handleChange}
                />
              </div>
            </div>

            {/* Message Display */}
            {msg && (
              <div className={`p-4 rounded-lg ${msg.includes("successful") ? "bg-green-500/10 border border-green-500/20 text-green-400" : "bg-red-500/10 border border-red-500/20 text-red-400"}`}>
                {msg}
              </div>
            )}

            {/* Submit Button */}
            <Button 
              type="submit" 
              className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              Create Vendor Account
            </Button>
          </form>

          {/* Footer Link */}
          <div className="mt-6 text-center">
            <span className="text-slate-400">Already have an account? </span>
            <a href="/login" className="text-blue-400 hover:text-blue-300 font-medium transition">
              Sign in here
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;