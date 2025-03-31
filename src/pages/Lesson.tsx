import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Check, X, ChevronRight, ChevronLeft, Tv, Loader2, LogIn } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useToast } from '@/hooks/use-toast';
import { lessonData } from '@/data/lessonData';
import { useProgress } from '@/contexts/ProgressContext';
import { useAuth } from '@/contexts/AuthContext';
import WebcamWithStreaming from '@/components/WebcamWithStreaming';
import { isCameraSupported } from '@/lib/webcam';

type LessonStatus = 'correct' | 'incorrect' | 'skipped' | null;

const Lesson = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { completeLesson, updateLetterAccuracy, loading: progressLoading } = useProgress();
  const { user } = useAuth();
  
  const [lesson, setLesson] = useState<any>(null);
  const [currentLetterIndex, setCurrentLetterIndex] = useState(0);
  const [status, setStatus] = useState<LessonStatus>(null);
  const [attempts, setAttempts] = useState(0);
  const [earnedXP, setEarnedXP] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [showWebcam, setShowWebcam] = useState(true);
  const [webcamKey, setWebcamKey] = useState(Date.now());
  const [webcamSupported, setWebcamSupported] = useState(true);
  const [latestPrediction, setLatestPrediction] = useState<any>(null);
  
  useEffect(() => {
    if (!id) {
      navigate('/lessons');
      return;
    }

    for (const category of Object.values(lessonData)) {
      const foundLesson = category.lessons.find(lesson => lesson.id === id);
      if (foundLesson) {
        setLesson(foundLesson);
        setCurrentLetterIndex(0);
        setStatus(null);
        setAttempts(0);
        setEarnedXP(0);
        setIsLoading(false);
        return;
      }
    }
    
    toast({
      title: "Lesson not found",
      description: "The requested lesson could not be found.",
      variant: "destructive"
    });
    navigate('/lessons');
  }, [id, navigate, toast]);

  useEffect(() => {
    setIsLoading(progressLoading);
  }, [progressLoading]);

  useEffect(() => {
    setWebcamSupported(isCameraSupported());
  }, []);

  useEffect(() => {
    if (!lesson || !latestPrediction || status !== null) return;
  
    const currentLetter = lesson.content[currentLetterIndex];
    if (
      latestPrediction.letter === currentLetter &&
      latestPrediction.confidence > 0.75
    ) {
      handleAnswerCheck(true);
    }
  }, [latestPrediction, lesson, currentLetterIndex, status]);

  // const handlePrediction = (prediction: any) => {
  //   setLatestPrediction(prediction);
    
  //   if (lesson && currentLetterIndex !== null && status === null) {
  //     const currentLetter = lesson.content[currentLetterIndex];
      
  //     if (prediction.letter === currentLetter && prediction.confidence > 0.75) {
  //       handleAnswerCheck(true);
  //     }
  //   }
  // };

  const handlePrediction = (prediction: any) => {
    setLatestPrediction(prediction);
  };

  const handleNextLetter = () => {
    if (!lesson) return;
    
    if (status === 'correct') {
      const baseXP = 5;
      const bonusXP = Math.max(0, 5 - attempts) * 2;
      const letterXP = baseXP + bonusXP;
      setEarnedXP(prev => prev + letterXP);
    }
    
    if (currentLetterIndex < lesson.content.length - 1) {
      setCurrentLetterIndex(currentLetterIndex + 1);
      setStatus(null);
      setAttempts(0);
    } else {
      if (user) {
        completeLesson(lesson.id, earnedXP > 0 ? earnedXP : lesson.xp);
      } else {
        toast({
          title: "Lesson completed!",
          description: `You earned ${earnedXP > 0 ? earnedXP : lesson.xp} XP, but progress isn't saved. Sign in to track your progress.`,
        });
      }
    }
  };

  const handlePreviousLetter = () => {
    if (currentLetterIndex > 0) {
      setCurrentLetterIndex(currentLetterIndex - 1);
      setStatus(null);
      setAttempts(0);
    }
  };

  const handleFinishLesson = () => {
    if (!lesson) return;
    
    let finalXP = earnedXP;
    
    if (status === 'correct') {
      const baseXP = 5;
      const bonusXP = Math.max(0, 5 - attempts) * 2;
      const letterXP = baseXP + bonusXP;
      finalXP += letterXP;
    }
    
    if (finalXP === 0) {
      finalXP = lesson.xp;
    }
    
    if (user) {
      completeLesson(lesson.id, finalXP);
    } else {
      toast({
        title: "Lesson completed!",
        description: `You earned ${finalXP} XP, but progress isn't saved. Sign in to track your progress.`,
      });
    }
    
    navigate('/lessons');
  };

  const handleAnswerCheck = (isCorrect: boolean) => {
    setAttempts(attempts + 1);
    
    if (isCorrect) {
      setStatus('correct');
      if (user) {
        const accuracy = Math.max(0, 100 - (attempts * 20));
        updateLetterAccuracy(lesson.content[currentLetterIndex], accuracy);
      }
    } else {
      setStatus('incorrect');
    }
  };

  const handleSkip = () => {
    setStatus('skipped');
    setAttempts(attempts + 1);
  };

  const toggleWebcam = () => {
    console.log("Toggling webcam from current state:", showWebcam ? "showing" : "hidden");
    if (!showWebcam) {
      setWebcamKey(Date.now());
      console.log("Webcam toggled to ON, new key:", Date.now());
    } else {
      console.log("Webcam toggled to OFF");
    }
    setShowWebcam(prev => !prev);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[70vh]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <p className="text-lg font-medium text-muted-foreground">Loading lesson...</p>
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="flex items-center justify-center h-[70vh]">
        <p>Lesson not found.</p>
      </div>
    );
  }

  const currentLetter = lesson.content[currentLetterIndex];
  const progress = ((currentLetterIndex + 1) / lesson.content.length) * 100;

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-6 animate-fade-in">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">{lesson?.title}</h1>
        <div className="flex items-center justify-between">
          <p className="text-muted-foreground">
            Practice signing each letter
          </p>
          <div className="flex items-center text-sm text-accent">
            <span>Earned XP: {earnedXP}</span>
          </div>
        </div>
      </div>

      {!user && (
        <Alert className="bg-muted/30 border-muted">
          <div className="flex items-center justify-between w-full">
            <AlertDescription className="text-muted-foreground">
              You're practicing as a guest. Sign in to save your progress!
            </AlertDescription>
            <Button variant="outline" size="sm" asChild>
              <a href="/auth/login">
                <LogIn className="mr-2 h-4 w-4" />
                Sign In
              </a>
            </Button>
          </div>
        </Alert>
      )}

      <Progress value={progress} className="h-2" />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="space-y-6">
          {showWebcam ? (
            webcamSupported ? (
              <WebcamWithStreaming key={webcamKey} onPrediction={handlePrediction} />
            ) : (
              <Card className="overflow-hidden">
                <CardContent className="p-6 flex items-center justify-center">
                  <div className="text-center">
                    <p className="text-sm text-destructive mb-2">
                      Your browser doesn't support webcam access.
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Try using a different browser like Chrome or Firefox.
                    </p>
                  </div>
                </CardContent>
              </Card>
            )
          ) : (
            <Card className="overflow-hidden">
              <CardContent className="p-6 flex items-center justify-center">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground mb-4">
                    For this demo, simply choose if you got it right or wrong:
                  </p>
                  <div className="flex space-x-4 justify-center">
                    <Button 
                      onClick={() => handleAnswerCheck(true)} 
                      variant="outline" 
                      className="flex-1 border-accent text-accent hover:bg-accent/10"
                      disabled={status !== null}
                    >
                      <Check className="h-5 w-5 mr-2" />
                      Got it right
                    </Button>
                    <Button 
                      onClick={() => handleAnswerCheck(false)} 
                      variant="outline" 
                      className="flex-1 border-destructive text-destructive hover:bg-destructive/10"
                      disabled={status !== null}
                    >
                      <X className="h-5 w-5 mr-2" />
                      Got it wrong
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Button 
            variant="outline" 
            className="w-full" 
            onClick={toggleWebcam}
            disabled={!webcamSupported && showWebcam}
          >
            {showWebcam ? "Show Demo Buttons" : "Show Webcam"}
          </Button>
        </div>

        <div className="space-y-6">
          <Card className="h-[200px] flex flex-col items-center justify-center">
            <CardContent className="flex flex-col items-center justify-center w-full h-full p-6">
              <div className="text-9xl font-bold mb-2">{currentLetter}</div>
              <p className="text-muted-foreground text-sm">
                Sign the letter shown above
              </p>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardContent className="p-4 flex items-center justify-center">
                <div className="flex flex-col items-center text-center">
                  <Tv className="h-6 w-6 mb-2 text-primary" />
                  <Button variant="outline" className="w-full" onClick={handleSkip}>
                    Skip for now
                  </Button>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 flex items-center justify-center">
                <div className="flex flex-col items-center text-center">
                  <div className="h-6 w-6 mb-2 flex items-center justify-center font-medium">
                    {currentLetterIndex + 1}/{lesson.content.length}
                  </div>
                  <Button 
                    onClick={status === null ? handleSkip : handleNextLetter} 
                    className="w-full"
                    variant={status === null ? "outline" : "default"}
                  >
                    {status === null ? "Don't know" : "Next"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {status && (
            <Alert variant={status === 'correct' ? 'default' : 'destructive'} className="animate-fade-in">
              <div className="flex items-center">
                {status === 'correct' ? (
                  <Check className="h-5 w-5 mr-2 text-accent" />
                ) : status === 'incorrect' ? (
                  <X className="h-5 w-5 mr-2" />
                ) : (
                  <ChevronRight className="h-5 w-5 mr-2" />
                )}
                <AlertDescription>
                  {status === 'correct' 
                    ? 'Great job! You got it right.' 
                    : status === 'incorrect'
                    ? `That wasn't quite right. Try practicing more. (Attempt ${attempts})`
                    : 'No problem, you can come back to this later.'}
                </AlertDescription>
              </div>
            </Alert>
          )}

          <div className="flex justify-between mt-6">
            <Button
              variant="outline"
              onClick={handlePreviousLetter}
              disabled={currentLetterIndex === 0}
            >
              <ChevronLeft className="h-4 w-4 mr-1" />
              Previous
            </Button>
            
            {currentLetterIndex === lesson.content.length - 1 ? (
              <Button onClick={handleFinishLesson}>
                Finish Lesson
              </Button>
            ) : (
              <Button 
                onClick={handleNextLetter}
                disabled={status === null}
              >
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Lesson;
