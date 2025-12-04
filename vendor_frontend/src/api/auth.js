import axios from "axios";

// Use relative URL so it works through the gateway proxy
const API = "/vendor-api";

export const loginUser = async (username, password) => {
  const form = new FormData();
  form.append("username", username);
  form.append("password", password);

  const res = await axios.post(`${API}/token`, form, {
    headers: { "Content-Type": "multipart/form-data" }
  });

  return res.data; // {access_token, token_type}
};

export const registerUser = async (username, password) => {
  const res = await axios.post(`${API}/register`, { username, password });
  return res.data;
};
