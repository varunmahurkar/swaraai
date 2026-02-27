
import { Link, useLocation } from 'react-router-dom';

export default function Header() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Profiles' },
    { path: '/generate', label: 'Generate' },
    { path: '/history', label: 'History' },
    { path: '/models', label: 'Models' },
  ];

  return (
    <header className="bg-gradient-to-r from-violet-700 via-purple-700 to-indigo-700 text-white shadow-xl">
      <nav className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-9 h-9 bg-white/20 rounded-xl flex items-center justify-center text-xl group-hover:bg-white/30 transition-colors">
            🎤
          </div>
          <span className="text-2xl font-bold tracking-tight">SwaraAI</span>
        </Link>

        <ul className="flex gap-1">
          {navItems.map(({ path, label }) => (
            <li key={path}>
              <Link
                to={path}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  location.pathname === path
                    ? 'bg-white/20 text-white'
                    : 'text-white/70 hover:bg-white/10 hover:text-white'
                }`}
              >
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
