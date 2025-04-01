
import React, { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Hand, BookOpen, BarChart, User, Menu, X, Loader2 } from 'lucide-react';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useIsMobile } from '@/hooks/use-mobile';
import { useAuth } from '@/contexts/AuthContext';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { supabase } from '@/integrations/supabase/client';

const navItems = [
  { name: 'Home', path: '/', icon: <Hand className="h-5 w-5" /> },
  { name: 'Lessons', path: '/lessons', icon: <BookOpen className="h-5 w-5" /> },
  { name: 'Progress', path: '/progress', icon: <BarChart className="h-5 w-5" /> },
  { name: 'Profile', path: '/profile', icon: <User className="h-5 w-5" /> },
];

const Navigation = () => {
  const location = useLocation();
  const isMobile = useIsMobile();
  const [open, setOpen] = React.useState(false);
  const { user, isLoading, signOut } = useAuth();
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);

  useEffect(() => {
    const fetchUserAvatar = async () => {
      if (user) {
        try {
          const { data, error } = await supabase
            .from('profiles')
            .select('avatar_url')
            .eq('id', user.id)
            .single();
            
          if (error) throw error;
          if (data) {
            setAvatarUrl(data.avatar_url);
          }
        } catch (error) {
          console.error('Error fetching user avatar:', error);
        }
      }
    };
    
    fetchUserAvatar();
    
    // Add listener for avatar updates
    const handleAvatarUpdate = (event: CustomEvent<string>) => {
      setAvatarUrl(event.detail);
    };
    
    window.addEventListener('avatar-updated', handleAvatarUpdate as EventListener);
    
    return () => {
      window.removeEventListener('avatar-updated', handleAvatarUpdate as EventListener);
    };
  }, [user]);

  const handleNavigation = () => {
    if (open) setOpen(false);
  };

  const getUserInitials = () => {
    if (!user) return 'G';
    
    // Try to get initials from user metadata if available
    const metadata = user.user_metadata;
    if (metadata) {
      const firstName = metadata.first_name || '';
      const lastName = metadata.last_name || '';
      if (firstName && lastName) {
        return `${firstName.charAt(0)}${lastName.charAt(0)}`;
      }
    }
    
    // Fallback to email
    return user.email ? user.email.charAt(0).toUpperCase() : 'U';
  };

  return (
    <header className="sticky top-0 z-40 border-b bg-card/80 backdrop-blur-sm">
      <div className="container mx-auto px-4 flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center space-x-2">
          <Hand className="h-8 w-8 text-accent" />
          <span className="text-xl font-bold tracking-tight text-primary">Show Me A Sign</span>
        </Link>

        {isMobile ? (
          <Sheet open={open} onOpenChange={setOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <Menu className="h-6 w-6" />
                <span className="sr-only">Toggle menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-60">
              <div className="flex flex-col gap-6 py-6">
                <nav className="flex flex-col gap-2">
                  {navItems.map((item) => (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={handleNavigation}
                      className={cn(
                        "flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium",
                        location.pathname === item.path
                          ? "bg-accent text-accent-foreground"
                          : "hover:bg-muted"
                      )}
                    >
                      {item.icon}
                      {item.name}
                    </Link>
                  ))}
                </nav>
                <div className="flex flex-col gap-2">
                  {isLoading ? (
                    <Button variant="ghost" disabled className="w-full">
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Loading...
                    </Button>
                  ) : user ? (
                    <>
                      <div className="flex items-center space-x-2 px-3 py-2">
                        <Avatar className="h-8 w-8">
                          <AvatarImage src={avatarUrl} alt="Avatar" />
                          <AvatarFallback>{getUserInitials()}</AvatarFallback>
                        </Avatar>
                        <div className="text-sm">
                          <p className="font-medium">{user.user_metadata?.first_name || user.email}</p>
                          <p className="text-xs text-muted-foreground">{user.email}</p>
                        </div>
                      </div>
                      <Button variant="outline" className="w-full" onClick={signOut}>Sign Out</Button>
                    </>
                  ) : (
                    <>
                      <Link to="/auth/login" onClick={handleNavigation}>
                        <Button variant="outline" className="w-full">Sign In</Button>
                      </Link>
                      <Link to="/auth/signup" onClick={handleNavigation}>
                        <Button className="w-full">Sign Up</Button>
                      </Link>
                    </>
                  )}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        ) : (
          <nav className="flex items-center gap-6">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  "flex items-center gap-1 text-sm font-medium transition-colors",
                  location.pathname === item.path
                    ? "text-accent"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {item.icon}
                {item.name}
              </Link>
            ))}
            <div className="flex items-center gap-2 ml-4">
              {isLoading ? (
                <Button variant="ghost" disabled>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Loading...
                </Button>
              ) : user ? (
                <div className="flex items-center gap-2">
                  <Link to="/profile">
                    <Avatar className="h-8 w-8 cursor-pointer">
                      <AvatarImage src={avatarUrl} alt="Avatar" />
                      <AvatarFallback>{getUserInitials()}</AvatarFallback>
                    </Avatar>
                  </Link>
                  <Button variant="outline" size="sm" onClick={signOut}>Sign Out</Button>
                </div>
              ) : (
                <>
                  <Link to="/auth/login">
                    <Button variant="outline">Sign In</Button>
                  </Link>
                  <Link to="/auth/signup">
                    <Button>Sign Up</Button>
                  </Link>
                </>
              )}
            </div>
          </nav>
        )}
      </div>
    </header>
  );
};

export default Navigation;
