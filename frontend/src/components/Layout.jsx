import logo from '../assets/chicken_logo_4x.png'

export default function Layout({ tabs, activeTab, onTabChange, children }) {
  return (
    <div className="h-screen flex flex-col bg-stone-100 text-stone-800">
      {/* Header */}
      <header className="shrink-0 bg-stone-800 border-b border-stone-700 px-6 py-4 flex items-center gap-3">
        <img src={logo} alt="ChickenCoopComfort logo" className="w-9 h-9 rounded-xl object-contain" />
        <div>
          <h1 className="text-base font-bold leading-none text-white tracking-tight">ChickenCoopComfort</h1>
          <p className="text-xs text-stone-400 mt-0.5">Smart coop monitoring & AI advisor</p>
        </div>
      </header>

      {/* Tabs */}
      <nav className="shrink-0 bg-stone-800 border-b border-stone-700 px-6 flex">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-amber-500 text-amber-400'
                : 'border-transparent text-stone-400 hover:text-stone-200'
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
