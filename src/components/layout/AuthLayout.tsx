
import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import { Hand } from 'lucide-react';

const AuthLayout = () => {
  return (
    <div className="min-h-screen flex flex-col justify-center items-center bg-background px-4">
      <div className="w-full max-w-md space-y-8">
        <div className="text-center">
          <Link to="/" className="inline-flex items-center space-x-2">
            <Hand size={36} className="text-accent" />
            <span className="text-3xl font-bold tracking-tight text-primary">Show Me A Sign</span>
          </Link>
          <h2 className="mt-6 text-2xl font-bold tracking-tight">Welcome to Show Me A Sign</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Learn American Sign Language interactively
          </p>
        </div>
        <Outlet />
      </div>
    </div>
  );
};

export default AuthLayout;
