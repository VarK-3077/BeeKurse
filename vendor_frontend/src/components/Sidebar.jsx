import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import { Menu } from "lucide-react";
import { Button } from "../components/ui/button";
import InputFiles from "./InputFiles";

const Sidebar = () => {
  return (
    <Collapsible defaultOpen className="h-screen flex flex-row-reverse border-r items-start">
      <div className="flex items-center justify-between p-2">
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="icon">
            <Menu className="h-5 w-5" />
          </Button>
        </CollapsibleTrigger>
      </div>
      
      <CollapsibleContent className="p-2 space-y-2">
        <InputFiles />
      </CollapsibleContent>
    </Collapsible>
  );
};

export default Sidebar;
