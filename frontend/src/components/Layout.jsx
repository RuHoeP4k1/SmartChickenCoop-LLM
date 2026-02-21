export default function Layout({ tabs, activeTab, onTabChange, children }) {
  return (
    <div className="h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="shrink-0 bg-slate-900 border-b border-slate-800 px-6 py-4 flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-green-700 flex items-center justify-center text-xl select-none">
          🐔
        </div>
        <div>
          <h1 className="text-base font-bold leading-none text-white tracking-tight">ChickenGuard</h1>
          <p className="text-xs text-slate-500 mt-0.5">Smart coop monitoring & AI advisor</p>
        </div>
      </header>

      {/* Tabs */}
      <nav className="shrink-0 bg-slate-900 border-b border-slate-800 px-6 flex">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-green-500 text-green-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {/* Page content */}
      <main className="flex-1 overflow-hidden">
        {children}
      </main>
    </div>
  )
}
