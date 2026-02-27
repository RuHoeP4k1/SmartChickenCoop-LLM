import logo from '../assets/chicken_logo_4x.png'

const QUICK_LINKS = [
  { id: 'chat',       icon: '💬', title: 'Chat with ChatKippieTee', desc: 'Ask anything about chicken health, feeding, or behaviour' },
  { id: 'sensors',    icon: '📡', title: 'Live Sensors',            desc: 'Real-time coop temperature, humidity, air quality & more' },
  { id: 'charts',     icon: '📈', title: 'Sensor Charts',           desc: 'Historical trends over the past hour, day, or week' },
  { id: 'alerts',     icon: '🔔', title: 'Alert Feed',              desc: 'Automated alerts when conditions need attention' },
  { id: 'eggs',       icon: '🥚', title: 'Egg Calendar',            desc: 'Track daily egg counts and view monthly totals' },
  { id: 'automation', icon: '⚙️', title: 'Automation',              desc: 'Door & ventilation automation status and controls' },
]

const FEATURES = [
  { icon: '🤖', title: 'AI-Powered Advice',     desc: 'Chat with an AI advisor trained on chicken-keeping knowledge. Get instant answers about health, feed, and coop management.' },
  { icon: '🌡️', title: 'Real-Time Monitoring',   desc: 'Keep track of temperature, humidity, H₂S levels, mold risk, feeder & waterer status — all updating live.' },
  { icon: '🚨', title: 'Smart Alerts',           desc: 'Automatic notifications when sensor readings hit warning or critical thresholds, with AI-generated action plans.' },
]

export default function Welcome({ onNavigate }) {
  return (
    <div className="h-full overflow-y-auto px-6 py-8 bg-stone-100">
      <div className="max-w-4xl mx-auto animate-fade-in">

        {/* Hero */}
        <div className="bg-stone-800 rounded-2xl px-8 py-10 mb-8 text-center">
          <img
            src={logo}
            alt="ChickenCoopComfort"
            className="w-20 h-20 mx-auto mb-4 rounded-2xl object-contain"
          />
          <h1 className="text-2xl font-bold text-white mb-2">
            Welcome to ChickenCoopComfort
          </h1>
          <p className="text-stone-400 text-sm leading-relaxed max-w-lg mx-auto mb-6">
            Smart coop monitoring powered by AI. Track conditions, chat with your coop assistant, and keep your flock happy and healthy.
          </p>
          <button
            onClick={() => onNavigate('chat')}
            className="bg-amber-600 hover:bg-amber-500 active:scale-95 text-white px-6 py-3 rounded-xl text-sm font-semibold transition-all duration-200"
          >
            Start Chatting
          </button>
        </div>

        {/* Quick-start cards */}
        <p className="text-xs font-semibold text-stone-400 uppercase tracking-widest mb-3">Quick Start</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {QUICK_LINKS.map(link => (
            <button
              key={link.id}
              onClick={() => onNavigate(link.id)}
              className="text-left rounded-2xl border border-stone-200 bg-white p-5 hover:shadow-md hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-200 cursor-pointer"
            >
              <span className="text-2xl mb-3 block">{link.icon}</span>
              <h3 className="text-sm font-semibold text-stone-800 mb-1">{link.title}</h3>
              <p className="text-xs text-stone-500 leading-relaxed">{link.desc}</p>
            </button>
          ))}
        </div>

        {/* Feature overview */}
        <p className="text-xs font-semibold text-stone-400 uppercase tracking-widest mb-3">Key Features</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {FEATURES.map(f => (
            <div
              key={f.title}
              className="rounded-2xl border border-stone-200 bg-white p-5"
            >
              <span className="text-2xl mb-3 block">{f.icon}</span>
              <h3 className="text-sm font-semibold text-stone-800 mb-1">{f.title}</h3>
              <p className="text-xs text-stone-500 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
