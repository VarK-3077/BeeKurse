import DisplayMessage from "./DisplayMessage";

const DisplayMessages = ({ Chats, messagesEndRef }) => {
  return (
    <div className="w-full flex flex-col gap-2 px-4">
      {Chats.map((chat, index) => (
        <DisplayMessage key={index} message={chat} />
      ))}

      <div ref={messagesEndRef} />
    </div>
  );
};

export default DisplayMessages;
