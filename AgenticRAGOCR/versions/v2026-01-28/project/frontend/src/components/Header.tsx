import { Settings, MessageCircle, HelpCircle, PhoneCall } from 'lucide-react';
import { Button } from './ui/button';

export function Header() {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm px-6 py-4 flex items-center justify-between">
      <h1 className="bg-gradient-to-r from-primary via-purple-500 to-accent bg-clip-text text-transparent">
        多模态文档解析智能体
      </h1>
      
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-accent transition-colors">
          <Settings className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-accent transition-colors">
          <MessageCircle className="h-5 w-5" />
        </Button>
        <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-accent transition-colors">
          <HelpCircle className="h-5 w-5" />
        </Button>
      </div>
    </header>
  );
}
