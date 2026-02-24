import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Cpu, Layers, Zap, Target, ArrowRight, Upload, MousePointer2, Download } from "lucide-react";
import LungWireframe from "@/components/LungWireframe";

const features = [
  { icon: Cpu, title: "AI-Powered", desc: "SAM-based auto segmentation with box prompts and auto-outline." },
  { icon: Layers, title: "Multi-Class", desc: "Segment multiple anatomical structures with distinct color overlays." },
  { icon: Zap, title: "Real-Time", desc: "Instant mask preview with paint, erase, and polygon tools." },
  { icon: Target, title: "Precise", desc: "Pixel-level accuracy with undo/redo and adjustable brush sizes." },
];

const steps = [
  { num: "01", icon: Upload, title: "Import", desc: "Upload your CT, MRI, or surgical video frames." },
  { num: "02", icon: MousePointer2, title: "Segment", desc: "Use AI box prompts or manual tools to create precise masks." },
  { num: "03", icon: Download, title: "Export", desc: "Save annotated masks for training or clinical review." },
];

// Generate deterministic particle positions
const particles = Array.from({ length: 20 }, (_, i) => ({
  left: `${(i * 17 + 7) % 100}%`,
  top: `${(i * 23 + 13) % 100}%`,
  size: 2 + (i % 4) * 2,
  delay: `${(i * 0.8) % 6}s`,
  duration: `${4 + (i % 3) * 2}s`,
}));

const Index = () => {
  return (
    <div className="dark min-h-screen bg-background text-foreground">
      {/* Nav */}
      <nav className="fixed top-0 w-full z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 h-16">
          <span className="font-mono font-extrabold text-lg tracking-widest text-primary">
            MEDSEG.IO
          </span>
          <div className="hidden md:flex items-center gap-8 font-mono text-sm text-muted-foreground">
            <a href="#features" className="hover:text-foreground transition-colors">Solutions</a>
            <a href="#how-it-works" className="hover:text-foreground transition-colors">Vision</a>
            <a href="#docs" className="hover:text-foreground transition-colors">Docs</a>
          </div>
          <div className="flex items-center gap-4">
            <span className="hidden sm:inline text-sm font-mono text-muted-foreground cursor-pointer hover:text-foreground transition-colors">
              Log In
            </span>
            <Link to="/tool">
              <Button size="sm" className="font-mono font-bold tracking-wide">
                Get Started
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
        {/* Video background */}
        <video
          autoPlay
          muted
          loop
          playsInline
          className="absolute inset-0 w-full h-full object-cover"
        >
          <source src="/vid.mp4" type="video/mp4" />
        </video>

        {/* Gradient overlay for text readability */}
        <div className="absolute inset-0 bg-gradient-to-r from-background/95 via-background/60 to-background/20" />

        {/* Particles */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {particles.map((p, i) => (
            <div
              key={i}
              className="particle absolute rounded-full bg-primary/30"
              style={{
                left: p.left,
                top: p.top,
                width: p.size,
                height: p.size,
                animationDelay: p.delay,
                animationDuration: p.duration,
              }}
            />
          ))}
        </div>

        {/* Lung Wireframe — subtle behind content on right */}
        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-[400px] h-[400px] lg:w-[500px] lg:h-[500px] opacity-30 pointer-events-none">
          <LungWireframe />
        </div>

        {/* Content */}
        <div className="relative z-10 w-full px-6 lg:pl-[8%] text-center lg:text-left">
          <div className="max-w-xl">
            <p className="font-mono text-xs tracking-[0.3em] text-primary mb-6 uppercase">
              Medical Image Analysis Platform
            </p>
            <h1 className="font-mono font-extrabold text-4xl sm:text-5xl md:text-6xl lg:text-7xl leading-[1.05] tracking-tight mb-6">
              PRECISION IN
              <br />
              <span className="text-primary">MEDICAL</span>
              <br />
              SEGMENTATION
            </h1>
            <p className="text-muted-foreground text-base sm:text-lg max-w-xl mb-10 font-sans leading-relaxed">
              Automated annotation for CTs, MRIs, and surgical video.
              AI-assisted segmentation with real-time mask editing, multi-class support, and SAM integration.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link to="/tool">
                <Button size="lg" className="font-mono font-bold tracking-wide text-base px-8 gap-2">
                  Get Started <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
              <Button
                variant="outline"
                size="lg"
                className="font-mono tracking-wide text-base px-8 border-border hover:bg-secondary"
              >
                View Docs
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="relative py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <p className="font-mono text-xs tracking-[0.3em] text-primary mb-4 text-center uppercase">
            Capabilities
          </p>
          <h2 className="font-mono font-bold text-3xl md:text-4xl text-center mb-16">
            Built for Medical Workflows
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((f, i) => (
              <div
                key={i}
                className="group p-6 rounded-lg border border-border bg-card/50 hover:border-primary/40 transition-all duration-300"
              >
                <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <f.icon className="w-5 h-5 text-primary" />
                </div>
                <h3 className="font-mono font-bold text-sm mb-2">{f.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section id="how-it-works" className="relative py-24 px-6 border-t border-border">
        <div className="max-w-5xl mx-auto">
          <p className="font-mono text-xs tracking-[0.3em] text-primary mb-4 text-center uppercase">
            Workflow
          </p>
          <h2 className="font-mono font-bold text-3xl md:text-4xl text-center mb-16">
            How It Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {steps.map((s, i) => (
              <div key={i} className="relative text-center group">
                {/* Connector line */}
                {i < steps.length - 1 && (
                  <div className="hidden md:block absolute top-10 left-[60%] w-[80%] h-px bg-gradient-to-r from-primary/30 to-transparent" />
                )}
                <div className="w-20 h-20 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto mb-6 group-hover:bg-primary/20 group-hover:border-primary/40 transition-all duration-300">
                  <s.icon className="w-8 h-8 text-primary" />
                </div>
                <span className="font-mono text-xs text-primary/60 tracking-widest">{s.num}</span>
                <h3 className="font-mono font-bold text-lg mt-1 mb-2">{s.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed max-w-xs mx-auto">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-6">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground tracking-widest">MEDSEG.IO</span>
          <span className="text-xs text-muted-foreground">© 2026 All rights reserved.</span>
        </div>
      </footer>
    </div>
  );
};

export default Index;
