import ReactMarkdown from "react-markdown";

const DisplayMessage = ({ message }) => {
  let styleAttr =
    "self-start text-white rounded-lg w-full [&_h1]:text-2xl [&_h1]:font-bold [&_h2]:text-xl [&_h2]:font-bold [&_h3]:text-lg [&_h3]:font-bold [&_p]:mb-2";

  if (message.role === "USER") {
    styleAttr =
      "self-end bg-slate-900 w-2/3 rounded-lg [&_h1]:text-2xl [&_h1]:font-bold [&_h2]:text-xl [&_h2]:font-bold [&_h3]:text-lg [&_h3]:font-bold [&_p]:mb-2";
  }

  return (
    <div className={styleAttr + " my-4 p-4"}>
      <ReactMarkdown>{message.content}</ReactMarkdown>
    </div>
  );
};

export default DisplayMessage;
