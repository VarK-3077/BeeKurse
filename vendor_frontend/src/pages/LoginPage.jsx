import { useState } from "react";
import { loginUser } from "../api/auth";
import { Button } from "../components/ui/button";
import { Store, User, Lock } from "lucide-react";

const LoginPage = () => {
  const [username, setUser] = useState("");
  const [password, setPass] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setErr("");
    setLoading(true);
    
    try {
      const data = await loginUser(username, password);
      localStorage.setItem("token", data.access_token);
      window.location.href = "/";
    } catch {
      setErr("Invalid username or password");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-linear-to-br from-blue-500 to-purple-600 mb-4">
            <Store className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">Welcome Back</h1>
          <p className="text-slate-400">Sign in to manage your vendor account</p>
        </div>

        {/* Login Card */}
        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl p-8 shadow-2xl">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Username
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-11 pr-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUser(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && submit(e)}
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-11 pr-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  placeholder="Enter your password"
                  type="password"
                  value={password}
                  onChange={(e) => setPass(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && submit(e)}
                />
              </div>
            </div>

            {err && (
              <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
                {err}
              </div>
            )}

            <Button 
              type="button"
              onClick={submit}
              disabled={loading}
              className="w-full bg-linear-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 rounded-lg transition-all duration-200 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Signing in..." : "Sign In"}
            </Button>
          </div>

          {/* Footer Link */}
          <div className="mt-6 text-center">
            <span className="text-slate-400">Don't have an account? </span>
            <a href="/register" className="text-blue-400 hover:text-blue-300 font-medium transition">
              Register as vendor
            </a>
          </div>
        </div>

        {/* Additional Info */}
        <div className="mt-6 text-center text-sm text-slate-500">
          <p>By signing in, you agree to our Terms of Service</p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;