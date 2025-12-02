import { createBrowserRouter, Navigate } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";

const isLoggedIn = () => {
  return !!localStorage.getItem("token");
};

const Protected = ({ children }) => {
  if (!isLoggedIn()) return <Navigate to="/login" replace />;
  return children;
};

export const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <Protected>
        <DashboardPage />
      </Protected>
    )
  },
  { path: "/login", element: <LoginPage /> },
  { path: "/register", element: <RegisterPage /> }
]);
