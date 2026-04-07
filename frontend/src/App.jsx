import { useState } from 'react'
import Layout from './components/Layout'
import Welcome from './components/Welcome'
import ChatPanel from './components/ChatPanel'
import SensorDashboard from './components/SensorDashboard'
import SensorChart from './components/SensorChart'
import AlertFeed from './components/AlertFeed'
import CoopLog from './components/CoopLog'
import AutomationPanel from './components/AutomationPanel'
import Weather from './components/Weather'
import MyChickens from './components/MyChickens'
import HeatmapView from './components/HeatmapView'
import Packages from './components/Packages'
import ResponseReview from './components/ResponseReview'

const TABS = [
  { id: 'welcome',    label: 'Home' },
  { id: 'chat',       label: 'Chat' },
  { id: 'sensors',    label: 'Sensors' },
  { id: 'heatmap',    label: 'Heatmap' },
  { id: 'charts',     label: 'Charts' },
  { id: 'alerts',     label: 'Alerts' },
  { id: 'review',     label: 'Review' },
  { id: 'eggs',       label: 'Coop Log' },
  { id: 'automation', label: 'Automation' },
  { id: 'weather',    label: 'Weather' },
  { id: 'chickens',   label: 'My Chickens' },
  { id: 'packages',   label: 'Packages' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('welcome')

  return (
    <Layout tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab}>
      <div className={activeTab === 'welcome'    ? 'h-full' : 'hidden'}><Welcome onNavigate={setActiveTab} /></div>
      <div className={activeTab === 'chat'       ? 'h-full' : 'hidden'}><ChatPanel /></div>
      <div className={activeTab === 'sensors'    ? 'h-full' : 'hidden'}><SensorDashboard /></div>
      <div className={activeTab === 'charts'     ? 'h-full' : 'hidden'}>
        <div className="h-full overflow-y-auto px-6 py-8 bg-stone-50 dark:bg-stone-900">
          <div className="max-w-4xl mx-auto animate-fade-in">
            <SensorChart />
          </div>
        </div>
      </div>
      <div className={activeTab === 'alerts'     ? 'h-full' : 'hidden'}><AlertFeed /></div>
      <div className={activeTab === 'review'     ? 'h-full' : 'hidden'}><ResponseReview /></div>
      <div className={activeTab === 'eggs'       ? 'h-full' : 'hidden'}><CoopLog /></div>
      <div className={activeTab === 'automation' ? 'h-full' : 'hidden'}><AutomationPanel /></div>
      <div className={activeTab === 'weather'    ? 'h-full' : 'hidden'}><Weather /></div>
      <div className={activeTab === 'chickens'   ? 'h-full' : 'hidden'}><MyChickens /></div>
      <div className={activeTab === 'packages'   ? 'h-full' : 'hidden'}><Packages /></div>
      <div className={activeTab === 'heatmap'    ? 'h-full' : 'hidden'}><HeatmapView /></div>
    </Layout>
  )
}
